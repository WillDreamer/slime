import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, RunConfig
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

# Set up logger for this module
logger = logging.getLogger(__name__)

# SGLang function-call parser to use when extracting <tool_call> blocks from the
# model's raw text. This MUST match the model's chat-template tool-call format:
#   - Qwen2.5 / Qwen3 (non-coder): JSON args inside <tool_call>...</tool_call>
#       -> "qwen25"
#   - Qwen3.5-4B-Base (and Qwen3-Coder): the template emits
#       <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
#       -> "qwen3_coder"
# Mismatch silently yields zero parsed tool calls (every turn becomes a RESPOND),
# which collapses training. Default to qwen3_coder for the Qwen3.5 migration;
# override with TAU_TOOL_PARSER if you swap base models.
TOOL_PARSER_TYPE = os.environ.get("TAU_TOOL_PARSER", "qwen3_coder")


# Format-check regexes for the per-turn shape we want every assistant turn to take:
#   (A) tool-call turn:  [<think>...</think>]{optional preamble}<tool_call>...</tool_call>
#   (B) respond turn:    [<think>...</think>]{plain natural-language reply}
# The `<think>...</think>` prefix is OPTIONAL: this model (TauSFT init) is a
# no-think tool-calling agent, and Qwen3's chat template strips `<think>` from
# every historical assistant turn before the last user query anyway, so the
# canonical per-turn shape carries no visible CoT. Requiring `<think>` made
# format_ok==0 on 100% of turns, turning FORMAT_BAD_PENALTY into a flat tax on
# winners (zero signal).
#
# IMPORTANT — natural-language PREAMBLE before <tool_call> is ALLOWED (the fix
# that mattered): the model's dominant correct shape is a short sentence ("Let me
# look up your order.") FOLLOWED by the call. The old _FORMAT_TOOLCALL_RE pinned
# <tool_call> to the very start (after optional <think>), so every "preamble +
# call" turn fell through to the respond branch, hit <tool_call> in the tail, and
# was scored format-bad. Measured on 52,482 real assistant turns from
# s3://whx-agent/AECE/tau_rl/tau_rl_Qwen35-4B_nothink_bs_32: 33% of ALL turns are
# this "preamble + one call" shape, and they were ALL mis-flagged — so format_ok
# sat at ~0.00-0.15 (NOT the ~96% an earlier comment claimed) and corr(num_tool_calls,
# format_ok) went NEGATIVE: a reward term that PENALIZED correct tool use. Allowing
# the preamble flips trajectory format_ok to ~0.58-0.81 with zero new false-OKs.
#
# We still reject genuine structural garbage: more than one <tool_call>, an
# orphan/unclosed </tool_call>, text AFTER </tool_call> (the rollout STOPs on
# </tool_call>, so a real tool turn ends exactly there — observed in 0/52482
# turns), leftover <tool_response>/<|im_*|> markup, or a malformed/double <think>.
# A tool turn's PREAMBLE is also scanned for that markup (only the user-visible
# reply text before the call may be free-form). Bad turns pay an additive
# FORMAT_BAD_PENALTY on task reward.
#
# _FORMAT_TOOLCALL_RE: optional <think>, then arbitrary preamble (captured so the
# caller can scan it for markup), then EXACTLY ONE <tool_call>...</tool_call> with
# nothing but whitespace after it. The inner `(?:(?!</tool_call>).)*` is a
# tempered-dot so the block can't swallow a second close tag.
_FORMAT_TOOLCALL_RE = re.compile(
    r"\A\s*(?:<think>(?!.*<think>).*?</think>\s*)?"
    r"(?P<pre>.*?)<tool_call>(?!.*<tool_call>)(?:(?!</tool_call>).)*</tool_call>\s*\Z",
    re.DOTALL,
)
_FORMAT_THINK_PREFIX_RE = re.compile(
    r"\A\s*(?:<think>(?!.*<think>).*?</think>)?(?P<after>.*)\Z",
    re.DOTALL,
)
# Markup that must never appear in free-form text (a respond turn's whole body, or
# a tool turn's preamble). Shared by both branches of _assistant_turn_format_ok.
_FORMAT_MARKUP_GARBAGE = (
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
)
# Additive penalties subtracted from total_reward when any assistant turn is
# format-bad. With the <think> requirement dropped AND the preamble-before-call
# shape allowed (see regex block), trajectory format_ok now passes on ~0.6-0.8 of
# rollouts (measured on the nothink run), so this fires on genuine structural
# garbage instead of on every correct tool turn — a real signal again, not an
# inverted tax that punished tool use. Weakened so format noise can't dominate
# the task signal:
#   - Successful (raw>0): pay 0.1 — light tap, still leaves +0.9 reward.
#   - Failed (raw==0): no penalty. We rely on the task gradient (and SFT) for
#     format learning; double-charging failures was hurting more than helping.
FORMAT_BAD_PENALTY_SUCCESS = 0.1
FORMAT_BAD_PENALTY_FAIL = 0.0

# Think length penalty is DISABLED. Earlier ablation showed it created a
# perverse gradient ("failed + long think" got -0.3, worse than truncation),
# pulling the model away from the SFT init's long-CoT distribution and
# collapsing tau-bench planning quality. We still measure avg_think_chars
# below for wandb monitoring, but the penalty term is forced to 0.
THINK_BUDGET_PER_TURN_CHARS = 500
THINK_PENALTY_PER_OVERAGE_CHAR = 0.0
THINK_PENALTY_CAP = 0.0
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _assistant_turn_format_ok(content: str) -> bool:
    """A single assistant turn is well-formed iff its raw text matches one of:
        [<think>...</think>]{optional preamble}<tool_call>...</tool_call>  (tool turn)
        [<think>...</think>]{plain text without XML markup garbage}        (respond turn)
    The `<think>...</think>` prefix is OPTIONAL (see regex block above), and a tool
    turn MAY open with natural-language preamble before the call. Reject
    double-<think>, more than one <tool_call>, an orphan/unclosed </tool_call>,
    text after </tool_call>, or leftover <|im_start|>/<|im_end|>/<tool_response>/
    <think> markup — in the respond body OR in a tool turn's preamble.
    """
    if not content:
        return False
    m = _FORMAT_TOOLCALL_RE.fullmatch(content)
    if m is not None:
        # Tool turn: the preamble (user-visible text before the call) may be
        # free-form, but must not smuggle in stray markup / a second think block.
        return not any(bad in m.group("pre") for bad in _FORMAT_MARKUP_GARBAGE)
    m = _FORMAT_THINK_PREFIX_RE.fullmatch(content)
    if m is None:
        return False
    after = m.group("after")
    return not any(bad in after for bad in _FORMAT_MARKUP_GARBAGE)


def _trajectory_format_ok(messages: list[dict[str, Any]]) -> tuple[bool, int, int]:
    """Returns (all_ok, num_assistant_turns, num_bad_turns)."""
    n_turns = 0
    n_bad = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        n_turns += 1
        if not _assistant_turn_format_ok(msg.get("content", "")):
            n_bad += 1
    return (n_bad == 0 and n_turns > 0), n_turns, n_bad


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: int | None = None
    status: Status = Status.COMPLETED
    rollout_log_probs: list[float] | None = None


def call_to_action_sglang(calls: list[Any], text_response: str) -> Action:
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    # Default action if no action was found.
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
    if calls:
        if len(calls) > 1:
            logger.debug("Multiple tool calls identified, only taking first.")
        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            logger.warning(f"{params} does not follow dict structure for action")
        else:
            action = Action(name=tool_call["name"], kwargs=params)
    return action


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class TrainableAgentMixin:
    """
    Mixin class that provides trainable agent functionality for tau-bench environments.

    This mixin extends the original tau-bench agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Inject the tau-bench tool-use guidance into the rendered chat-template
        prompt (one-or-no call per turn / MUST execute tools / NEVER pre-confirm).

        IMPORTANT — template-format dependent:
          * Qwen3 / Hermes templates carry the sentence
                "You may call one or more functions to assist with the user query."
            which we replace outright (tau allows at most ONE call per turn).
          * Qwen3.5-4B-Base / Qwen3-Coder use the XML tool format and DO NOT
            contain that sentence at all — its tools block reads
                "If you choose to call a function ONLY reply in the following
                 format with NO suffix: <tool_call><function=...>..."
            so the old `.replace(...)` was a NO-OP there and the tau guidance was
            silently dropped from every prompt. For that template we insert
            TOOL_INSTRUCTION just before the format spec instead.

        Applied identically at sampling and training render time (both go through
        _render_messages_text), so the prompt prefix stays byte-consistent and
        the assistant-boundary alignment in _build_training_tensor is unaffected.
        """
        # Idempotency guard: render output is always fresh template text (never
        # contains TOOL_INSTRUCTION), so this normally no-ops; the guard just
        # prevents a double-injection if this is ever called on already-patched
        # text (the qwen3.5 branch keeps the anchor line, so a naive re-run would
        # inject twice and desync the sampling/training prefixes).
        if TOOL_INSTRUCTION.strip() in text:
            return text

        qwen3_anchor = "You may call one or more functions to assist with the user query."
        if qwen3_anchor in text:
            return text.replace(qwen3_anchor, TOOL_INSTRUCTION)

        # Qwen3.5 / qwen3-coder XML template: inject before the format spec line.
        qwen35_anchor = "If you choose to call a function ONLY reply in the following format"
        if qwen35_anchor in text:
            return text.replace(qwen35_anchor, TOOL_INSTRUCTION.strip() + "\n\n" + qwen35_anchor, 1)

        # Neither anchor present (template drift): warn ONCE, leave prompt intact
        # rather than crash. Tool parsing can still work; only the behavioral
        # guidance is absent, so this is a soft degradation, not a hard failure.
        if not getattr(self, "_warned_no_tool_anchor", False):
            logger.warning(
                "tau-bench TOOL_INSTRUCTION anchor not found in chat template; "
                "guidance not injected (tool-call format itself is unaffected). "
                "Check the model's chat_template if reward looks degraded."
            )
            self._warned_no_tool_anchor = True
        return text

    async def _call_llm(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Make an LLM call tracking.

        Args:
            url: SGLang server URL
            payload: Request payload containing text and sampling parameters
            headers: Optional HTTP headers (e.g. X-SMG-Routing-Key for session
                affinity when the router policy is consistent_hashing)

        Returns:
            LLM response from sglang server
        """
        return await post(url, payload, headers=headers)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from sglang

        Returns:
            Parsed tool call result in OpenAI format
        """
        return self.openai_adapter.parse_response_to_openai_format(response)

    async def _execute_tool(self, env, action: Action):
        """
        Execute a tool/action in the environment.

        Args:
            env: Tau-bench environment instance
            action: Action to execute

        Returns:
            Environment step result
        """
        return await env.step(action)

    async def _initialize_environment(self, env, task_index: int | None) -> tuple[str, dict[str, Any]]:
        """
        Initialize the environment and get initial observation.

        Args:
            env: Tau-bench environment instance
            task_index: Task index to reset to

        Returns:
            Tuple of (observation, info)
        """
        if task_index is not None:
            env_reset_res = await env.reset(task_index=task_index)
        else:
            env_reset_res = await env.reset()
        return env_reset_res.observation, env_reset_res.info.model_dump()

    def _build_initial_messages(self, obs: str) -> list[dict[str, Any]]:
        """
        Build initial conversation messages.

        Args:
            obs: Initial observation from environment

        Returns:
            List of initial messages
        """
        return [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]

    def _render_messages_text(
        self,
        state: GenerateState,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        """Render messages via apply_chat_template + tau-bench tool-instruction patch.

        enable_thinking=False is CRITICAL for Qwen3.5 (and any Qwen "thinking"
        model). Its chat template, with add_generation_prompt=True and thinking
        ON (the default), appends a LONE opening `<think>\\n` to the prompt — so
        the opening tag lives in the PROMPT, the model's sampled response is
        `reasoning…</think>…` (an ORPHAN close), and _build_training_tensor's
        think-preserved / think-stripped alignment matches neither -> ~2/3 of
        response tokens get loss_mask=0 and format_ok==0 on every turn, which
        distorted the gradient and made raw_reward DROP over training (observed
        on tau_rl_Qwen35-4B: 74.6%->58.2% while response_len grew). With
        enable_thinking=False the template instead emits a CLOSED `<think>\\n\\n
        </think>\\n\\n` and the agent runs no-think (matching the original
        tau-bench design: a no-think tool-calling agent). Passed on BOTH the
        sampling render (add_generation_prompt=True) and the training render
        (add_generation_prompt=False) since both go through here, keeping the
        per-token prefix byte-consistent.

        Some HF tokenizers don't accept enable_thinking; pass it via a try/except
        so this stays safe on non-Qwen templates.
        """
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=self.tools_info,
            enable_thinking=False,
        )
        try:
            text = state.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Template/tokenizer doesn't support enable_thinking — fall back.
            kwargs.pop("enable_thinking", None)
            text = state.tokenizer.apply_chat_template(messages, **kwargs)
        return self._reformulate_tool_call(text)

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        """Render the initial prompt and return (text, token_ids)."""
        prompt_text = self._render_messages_text(state, messages, add_generation_prompt=True)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    @staticmethod
    def _find_subsequence_starts(haystack: list[int], needle: list[int]) -> list[int]:
        """Return all start positions of `needle` in `haystack`, non-overlapping greedy."""
        if not needle:
            return []
        out: list[int] = []
        nl = len(needle)
        i = 0
        while i <= len(haystack) - nl:
            if haystack[i:i + nl] == needle:
                out.append(i)
                i += nl
            else:
                i += 1
        return out

    def _build_training_tensor(
        self,
        state: GenerateState,
        messages: list[dict[str, Any]],
        prompt_token_ids: list[int],
        turn_samples: list[dict[str, Any]],
        im_end_id: int,
        asst_prompt_token_ids: list[int],
    ) -> tuple[list[int], list[int], list[float], str]:
        """Render the full trajectory once and rebuild (response, mask, logprob).

        Strategy:
          1. apply_chat_template(messages, ag=False) → final canonical text/tokens.
             This is what slime backprops on; per-token prefix at training time
             matches what each turn saw at sampling time (Qwen3's last_query_index
             logic strips historical <think> after the latest real user turn).
          2. Locate every `<|im_start|>assistant\\n` boundary in the rendered tokens
             and pair the i-th boundary with turn_samples[i].
          3. For each pair, the rendered span is one of:
               a) full sample (think preserved — turn is after last_query_index):
                  span == sampled_token_ids → mask=1, use sampled logprobs as-is
               b) post-think suffix (think stripped — turn is before last_query_index):
                  span == sampled_token_ids[k:] where k counts <think>...</think>\\n\\n
                  → mask=1 only on surviving tokens, with their post-think logprobs
               b') no-think render-extra (enable_thinking=False): span has an extra
                  leading empty <think>\\n\\n</think>\\n\\n that the sample lacks
                  (template injects it into the prompt + re-adds on the last turn);
                  sampled == span[lead:] → mask=1 on the tail (sampled) tokens
               c) neither (e.g., format-bad output, length-truncated final turn whose
                  </think> never closed, or whitespace de-canonicalization):
                  log a warning and leave mask=0 for the turn

        The trailing <|im_end|> of every well-formed turn is force-set to mask=1
        with logprob=0.0 so the policy keeps learning to terminate turns.
        """
        if not turn_samples:
            return [], [], [], ""

        final_text = self._render_messages_text(state, messages, add_generation_prompt=False)
        final_token_ids = state.tokenizer(final_text, add_special_tokens=False)["input_ids"]

        if final_token_ids[:len(prompt_token_ids)] != prompt_token_ids:
            logger.warning(
                "Final render does not start with the cached prompt prefix; "
                "tools list or system content drifted between renders."
            )

        response_token_ids = list(final_token_ids[len(prompt_token_ids):])
        loss_masks = [0] * len(response_token_ids)
        rollout_log_probs = [0.0] * len(response_token_ids)

        asst_starts = self._find_subsequence_starts(final_token_ids, asst_prompt_token_ids)
        # Each start[i] points at the position of `<|im_start|>` itself; the
        # assistant content begins +len(asst_prompt_token_ids) tokens later.
        asst_content_starts = [s + len(asst_prompt_token_ids) for s in asst_starts]

        # If the trajectory aborted/length-truncated mid-assistant, messages[-1] is
        # the broken assistant turn. Its rendered form may include a synthetic
        # `<think>\\n\\n</think>\\n\\n` wrapper if `</think>` never closed, which
        # shifts every token in the span and breaks alignment. Skip its training
        # signal entirely rather than backprop on garbage.
        last_msg_is_assistant = bool(messages) and messages[-1].get("role") == "assistant"

        if len(asst_content_starts) != len(turn_samples):
            logger.warning(
                f"Assistant boundary count {len(asst_content_starts)} != turn_samples count "
                f"{len(turn_samples)}; pairing by position with the smaller of the two."
            )

        n_pairs = min(len(asst_content_starts), len(turn_samples))
        for span_idx in range(n_pairs):
            sample = turn_samples[span_idx]
            sampled_tokens: list[int] = sample["sampled_token_ids"]
            sampled_logprobs: list[float] = sample["sampled_log_probs"]
            response_text: str = sample["response_text"]

            if last_msg_is_assistant and span_idx == n_pairs - 1:
                # Skip the broken trailing turn — its render is unreliable.
                continue

            content_start = asst_content_starts[span_idx]
            end = content_start
            while end < len(final_token_ids) and final_token_ids[end] != im_end_id:
                end += 1
            if end >= len(final_token_ids):
                logger.warning(f"Turn {span_idx}: no <|im_end|> after asst_start; skipping")
                continue

            span_tokens = final_token_ids[content_start:end]
            resp_content_start = content_start - len(prompt_token_ids)
            resp_im_end_pos = end - len(prompt_token_ids)
            if resp_content_start < 0 or resp_im_end_pos >= len(response_token_ids):
                logger.warning(f"Turn {span_idx}: response indices out of range; skipping")
                continue

            # (a) think preserved — direct match
            if span_tokens == sampled_tokens:
                for off, lp in enumerate(sampled_logprobs):
                    loss_masks[resp_content_start + off] = 1
                    rollout_log_probs[resp_content_start + off] = lp
                loss_masks[resp_im_end_pos] = 1
                continue

            # (b) think stripped — try post-think suffix match
            end_tag = response_text.find("</think>")
            if end_tag >= 0:
                after_close = end_tag + len("</think>")
                while after_close < len(response_text) and response_text[after_close] == "\n":
                    after_close += 1
                stripped_prefix_text = response_text[:after_close]
                stripped_prefix_tokens = state.tokenizer(
                    stripped_prefix_text, add_special_tokens=False
                )["input_ids"]
                n_stripped = len(stripped_prefix_tokens)
                if n_stripped <= len(sampled_tokens):
                    post_think_tokens = sampled_tokens[n_stripped:]
                    post_think_logprobs = sampled_logprobs[n_stripped:]
                    if span_tokens == post_think_tokens:
                        for off, lp in enumerate(post_think_logprobs):
                            loss_masks[resp_content_start + off] = 1
                            rollout_log_probs[resp_content_start + off] = lp
                        loss_masks[resp_im_end_pos] = 1
                        continue

            # (b') no-think (enable_thinking=False): the RENDER has an extra empty
            # `<think>\n\n</think>\n\n` block that the SAMPLE lacks — this is the
            # inverse of (b). With enable_thinking=False the template injects the
            # closed think block into the generation prompt, so the model never
            # samples it, but apply_chat_template re-adds it to the LAST assistant
            # turn (loop.index0 > last_query_index). So span = [empty-think] +
            # sampled. Match sampled_tokens against the TAIL of span_tokens and
            # mask only those (the leading empty-think tokens carry no sampled
            # logprob, so they stay mask=0).
            if len(span_tokens) > len(sampled_tokens) and len(sampled_tokens) > 0:
                lead = len(span_tokens) - len(sampled_tokens)
                if span_tokens[lead:] == sampled_tokens:
                    for off, lp in enumerate(sampled_logprobs):
                        loss_masks[resp_content_start + lead + off] = 1
                        rollout_log_probs[resp_content_start + lead + off] = lp
                    loss_masks[resp_im_end_pos] = 1
                    continue

            # (c) neither — alignment failed; leave mask=0
            logger.warning(
                f"Turn {span_idx}: render/sample alignment failed "
                f"(span_len={len(span_tokens)}, sampled_len={len(sampled_tokens)}); "
                f"no training signal for this turn."
            )

        response_text_decoded = state.tokenizer.decode(response_token_ids) if response_token_ids else ""
        return response_token_ids, loss_masks, rollout_log_probs, response_text_decoded

    @staticmethod
    def _build_verbose_response_text(messages: list[dict[str, Any]]) -> str:
        """Render the post-prompt suffix of `messages` preserving every turn's
        `<think>` block — saved to `res.response` for inspection only.

        The canonical apply_chat_template render (used for training tensors)
        strips historical `<think>` per Qwen3's last_query_index logic, so the
        saved response text loses CoT for every turn before the final user
        message. This helper bypasses the template and emits each message in
        Qwen's exact wire format, leaving assistant content (which already
        includes <think>...</think>) untouched.

        messages[0] is system, messages[1] is the initial user; both render
        into the prompt prefix. The response begins immediately after the
        prompt's trailing `<|im_start|>assistant\\n`, so the first assistant
        turn here is emitted *without* its own `<|im_start|>assistant\\n` —
        same convention as the existing canonical-stripped response_text.
        """
        parts: list[str] = []
        for i in range(2, len(messages)):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "") or ""
            if role == "assistant":
                if i == 2:
                    parts.append(f"{content}<|im_end|>\n")
                else:
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
            elif role == "tool":
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                )
            else:  # "user"
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        return "".join(parts)

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction for training.

        Per turn we re-render the full message list via apply_chat_template
        (so tool messages get the canonical `<|im_start|>user\\n<tool_response>...`
        wrap, and Qwen3's last_query_index logic strips historical <think> blocks
        once a real user turn arrives via the user simulator). The training
        tensor is rebuilt at end-of-trajectory from the same final render, with
        per-turn sampled logprobs aligned back onto the surviving tokens — see
        `_build_training_tensor` for the alignment cases.

        Trade-off vs. the prior string-concat path: a per-turn re-render costs
        one tokenizer pass on a bounded message list, but in exchange the
        rollout context exactly matches the SFT distribution and stops growing
        the trajectory's <think> footprint after every RESPOND turn.
        """
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"

        # Session-affinity routing for this multi-turn trajectory. When the router
        # policy is consistent_hashing, every turn of THIS trajectory must carry
        # the SAME routing key so the router hashes them all to one worker, which
        # then reuses its prefix cache across turns (the bulk of a tau trajectory's
        # prompt is the unchanged conversation history). The key is generated ONCE
        # here — not per turn — so it is stable for the whole `for` loop below.
        # tau drives the actor via this custom path (not slime's sglang_rollout),
        # so the X-SMG-Routing-Key header that slime normally sets must be set here
        # too; gate it on the same `router_policy == "consistent_hashing"` so any
        # other policy (e.g. the default cache_aware) is unchanged (headers=None).
        routing_headers = None
        if getattr(rollout_args, "router_policy", None) == "consistent_hashing":
            routing_headers = {"X-SMG-Routing-Key": uuid.uuid4().hex}

        im_end_id = state.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Tokenize the assistant prompt-prefix once for boundary scanning at
        # tensor-build time. On Qwen3 this is [151644, 77091, 198].
        asst_prompt_token_ids = state.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

        # Stop the moment the model closes a tool call OR ends the assistant
        # turn. Qwen3-Base's generation_config sets eos=<|endoftext|> (151643),
        # so sglang by default does NOT stop on <|im_end|> (151645). Without
        # an explicit stop on <|im_end|>, a RESPOND-style reply (natural-
        # language message to the user, no <tool_call>) burns through
        # max_new_tokens after emitting its own <|im_end|> — the model keeps
        # hallucinating a second <think> block past turn end.
        stop_list = list(sampling_params.get("stop") or [])
        if "</tool_call>" not in stop_list:
            stop_list = stop_list + ["</tool_call>"]
        stop_token_ids = list(sampling_params.get("stop_token_ids") or [])
        if im_end_id not in stop_token_ids:
            stop_token_ids = stop_token_ids + [im_end_id]
        sampling_params = {
            **sampling_params,
            "stop": stop_list,
            "stop_token_ids": stop_token_ids,
            "no_stop_trim": True,
        }

        # env.reset triggers the user-simulator LLM call, which can raise
        # transient network errors (httpcore.ReadError, etc.) that tau-bench's
        # user.py retry loop does not catch. Guard it here so one flaky sample
        # doesn't take down the whole asyncio.gather in generate_and_rm_group.
        try:
            obs, info = await self._initialize_environment(env, task_index)
        except Exception as e:
            logger.warning(f"env.reset failed for task {task_index}: {e.__class__.__name__}: {e}")
            reset_info = {
                "reset_error": str(e),
                "num_turns": 0,
                "num_tool_calls": 0,
                "num_respond_turns": 0,
                "num_length_trunc": 0,
                "has_tool_call": 0,
                "tool_call_turn_frac": 0.0,
                "answer_correct": 0,
                "env_done": 0,
                "is_aborted": 1,
            }
            res = InteractionResult(prompt="", reward=0, messages=[], info=reset_info)
            res.status = Status.ABORTED
            return self._build_final_result(res, 0.0, reset_info, [], [], [], [], [])

        messages = self._build_initial_messages(obs)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        total_reward = 0.0

        # Per-trajectory metrics — surfaced via info so slime's
        # compute_metrics_from_samples auto-logs them to wandb.
        num_turns = 0
        num_tool_calls = 0
        num_respond_turns = 0
        num_length_trunc = 0

        # Per-turn sample records for tensor reconstruction. We never use these
        # to drive inference (which always re-renders messages), only to map
        # sampled token-level logprobs back onto the final canonical render.
        turn_samples: list[dict[str, Any]] = []

        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})
        res.status = Status.TRUNCATED

        for _ in range(max_num_steps):
            # Re-render the current message list each turn. After a RESPOND turn
            # the user simulator's reply advances last_query_index; the chat
            # template then drops `<think>` from every earlier assistant turn,
            # so context grows sub-linearly in the number of tool calls.
            input_text = self._render_messages_text(state, messages, add_generation_prompt=True)

            payload = {
                "text": input_text,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            try:
                output = await self._call_llm(url, payload, headers=routing_headers)
            except Exception as e:
                logger.warning(
                    f"sglang HTTP call failed for task {task_index}: {e.__class__.__name__}: {e}"
                )
                res.status = Status.ABORTED
                break

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                res.status = Status.ABORTED
                break

            # Pull per-token ids + logprobs straight from sglang so every
            # trained token has the exact logprob that produced it.
            token_logprobs = output["meta_info"].get("output_token_logprobs") or []
            cur_token_ids = [item[1] for item in token_logprobs]
            cur_log_probs = [item[0] for item in token_logprobs]
            cur_response = output["text"]

            # Some sglang builds emit a trailing <|im_end|>; drop it so we can
            # append our own turn-terminator consistently below.
            while cur_token_ids and cur_token_ids[-1] == im_end_id:
                cur_token_ids.pop()
                cur_log_probs.pop()
            if cur_response.endswith("<|im_end|>"):
                cur_response = cur_response[: -len("<|im_end|>")]

            msg_idx = len(messages)
            messages.append({"role": "assistant", "content": cur_response})
            num_turns += 1
            turn_samples.append({
                "msg_idx": msg_idx,
                "sampled_token_ids": list(cur_token_ids),
                "sampled_log_probs": list(cur_log_probs),
                "response_text": cur_response,
                "finish_type": finish_type,
            })

            # If sglang hit max_new_tokens we intentionally stop here instead
            # of handing a half-written message to the user simulator — that
            # feedback loop is what produced the word-salad responses before.
            # The truncated turn's training signal is later dropped by
            # _build_training_tensor (its render is unreliable when </think>
            # never closed).
            if finish_type == "length":
                num_length_trunc += 1
                res.status = Status.TRUNCATED
                break

            try:
                openai_result = self._parse_tool(cur_response)
                if not openai_result["success"]:
                    logger.warning(
                        f"Tool parser failed: {openai_result['error']}; response[:200]={cur_response[:200]!r}"
                    )
                    res.status = Status.ABORTED
                    break
                parsed = openai_result["parsed_result"]
            except Exception as e:
                logger.warning(f"Exception in tool parser: {e}; response[:200]={cur_response[:200]!r}")
                res.status = Status.ABORTED
                break

            agent_content, calls = parsed["normal_text"], parsed["calls"]
            action = call_to_action_sglang(calls, agent_content)

            try:
                env_response = await self._execute_tool(env, action)
            except Exception as e:
                logger.warning("Environment step failed (typically a user simulator call error).")
                logger.warning(f"Error: {e}")
                res.status = Status.ABORTED
                break

            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                num_tool_calls += 1
                messages.append({
                    "role": "tool",
                    "name": action.name,
                    "content": env_response.observation,
                })
            else:
                num_respond_turns += 1
                messages.append({"role": "user", "content": env_response.observation})

            if env_response.done:
                res.status = Status.COMPLETED
                break

        # Reconstruct (response_token_ids, loss_masks, rollout_log_probs)
        # from the canonical final render. See _build_training_tensor for the
        # think-preserved / think-stripped alignment cases.
        response_token_ids, loss_masks, rollout_log_probs, response_text = self._build_training_tensor(
            state, messages, prompt_token_ids, turn_samples, im_end_id, asst_prompt_token_ids,
        )

        # Per-turn format check: every assistant turn must look like
        #   [<think>...</think>]<tool_call>...</tool_call>   or
        #   [<think>...</think>]{plain text}
        # (the <think> prefix is optional — this is a no-think agent and the
        # chat template strips historical <think> anyway; see regex block).
        # If any turn fails, subtract an additive penalty from the task reward.
        # We charge winners more than losers (see FORMAT_BAD_PENALTY_* above):
        # the old multiplicative form left 0-reward failures unpunished, so we
        # still bill them, but lightly so format noise doesn't drown the task
        # signal in the (much larger) failure bucket.
        format_ok, n_assist_turns, n_bad_turns = _trajectory_format_ok(messages)
        raw_task_reward = total_reward
        if not format_ok:
            penalty = FORMAT_BAD_PENALTY_SUCCESS if raw_task_reward > 0 else FORMAT_BAD_PENALTY_FAIL
            total_reward = total_reward - penalty

        # Direct CoT-length penalty: discourage verbose <think> blocks even
        # when the trajectory completes inside the length cap. Without this,
        # the model can keep growing CoT as long as it eventually finishes,
        # and the indirect "truncation = -0.2" signal only fires at the cliff.
        total_think_chars = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for blk in _THINK_BLOCK_RE.findall(msg.get("content", "")):
                total_think_chars += len(blk)
        avg_think_chars = total_think_chars / max(n_assist_turns, 1)
        think_overage = max(0.0, avg_think_chars - THINK_BUDGET_PER_TURN_CHARS)
        think_length_penalty = min(THINK_PENALTY_CAP, think_overage * THINK_PENALTY_PER_OVERAGE_CHAR)
        total_reward = total_reward - think_length_penalty

        # Populate the metadata keys slime auto-logs to wandb (see
        # slime/ray/rollout.py::compute_metrics_from_samples).
        info["num_turns"] = num_turns
        info["num_tool_calls"] = num_tool_calls
        info["num_respond_turns"] = num_respond_turns
        info["num_length_trunc"] = num_length_trunc
        info["has_tool_call"] = int(num_tool_calls > 0)
        info["tool_call_turn_frac"] = (num_tool_calls / num_turns) if num_turns > 0 else 0.0
        info["answer_correct"] = int(raw_task_reward > 0)
        info["env_done"] = int(res.status == Status.COMPLETED)
        info["is_aborted"] = int(res.status == Status.ABORTED)
        info["format_ok"] = int(format_ok)
        info["format_bad_turns"] = n_bad_turns
        info["raw_task_reward"] = float(raw_task_reward)
        info["avg_think_chars"] = float(avg_think_chars)
        info["think_length_penalty"] = float(think_length_penalty)
        info["trained_token_count"] = int(sum(loss_masks))

        # Saved `response` keeps every turn's <think> for inspection; training
        # tensors (tokens/loss_mask/rollout_log_probs) remain aligned with the
        # canonical stripped render so the per-token prefix at backprop time
        # still matches what each turn saw at sampling time.
        verbose_response_text = self._build_verbose_response_text(messages)
        return self._build_final_result(
            res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids,
            rollout_log_probs, response_text=verbose_response_text,
        )

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        rollout_log_probs: list[float] | None = None,
        response_text: str | None = None,
    ) -> InteractionResult:
        """
        Build the final interaction result with all collected data.

        Args:
            res: InteractionResult instance to populate
            total_reward: Total reward accumulated during interaction
            info: Environment info dictionary
            messages: Complete conversation messages
            loss_masks: Loss masks for training
            prompt_token_ids: Prompt token IDs
            response_token_ids: Response token IDs
            rollout_log_probs: Per-token log probs for TIS (agent tokens real, env tokens 0.0)

        Returns:
            Populated InteractionResult
        """
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        # response is the verbose (think-preserving) multi-turn rendered string
        # for inspection — built by _build_verbose_response_text. Note: it does
        # NOT match tokens token-for-token, because tokens come from the
        # canonical apply_chat_template render which strips historical <think>.
        # Training-relevant fields (tokens/loss_mask/rollout_log_probs) are the
        # source of truth; `response` is for vis_rollout_data / debugging only.
        res.response = response_text if response_text is not None else ""
        res.response_length = len(loss_masks)
        res.rollout_log_probs = rollout_log_probs

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"response_loss_mask_len={len(loss_masks)}, "
            f"prompt_token_len={len(prompt_token_ids)}, "
            f"response_token_len={len(response_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.

    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        # Initialize the parent ToolCallingAgent
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # Store rollout and sampling parameters as instance variables
        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": self.temperature,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
        }
        # Initialize OpenAI adapter
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type=TOOL_PARSER_TYPE)


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> Agent:
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
