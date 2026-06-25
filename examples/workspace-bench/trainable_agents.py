import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from ws_bench.agents.base import Agent
from ws_bench.agents.tool_calling_agent import ToolCallingAgent
from ws_bench.types import Action, RESPOND_ACTION_NAME, RunConfig
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

# Set up logger for this module
logger = logging.getLogger(__name__)

# The terminate tool whose step triggers the rubric judge (reward). Used for
# rollout time-attribution (which env.step blocked on the Bedrock judge call).
FINISH_TOOL_NAME = "finish"

# SGLang function-call parser to use when extracting <tool_call> blocks from the
# model's raw text. This MUST match the model's chat-template tool-call format:
#   - Qwen2.5 / Qwen3 (non-coder): JSON args inside <tool_call>...</tool_call>
#       -> "qwen25"
#   - Qwen3.5-4B-Base (and Qwen3-Coder): the template emits
#       <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
#       -> "qwen3_coder"
# Mismatch silently yields zero parsed tool calls (every turn becomes a RESPOND),
# which collapses training. Default to qwen3_coder for the Qwen3.5 migration;
# override with WS_TOOL_PARSER if you swap base models.
TOOL_PARSER_TYPE = os.environ.get("WS_TOOL_PARSER", "qwen3_coder")


# Format-check regexes for the per-turn shape we want every assistant turn to take:
#   (A) tool-call turn:  [<think>...</think>]{optional preamble}<tool_call>...</tool_call>
#   (B) respond turn:    [<think>...</think>]{plain natural-language reply}
# The `<think>...</think>` prefix is OPTIONAL: this model (Qwen3.5-4B-Base init)
# is a no-think tool-calling agent, and Qwen3's chat template strips `<think>`
# from every historical assistant turn before the last user query anyway, so the
# canonical per-turn shape carries no visible CoT. Requiring `<think>` made
# format_ok==0 on 100% of turns, turning FORMAT_BAD_PENALTY into a flat tax on
# winners (zero signal). (See the tau-bench migration notes — this regex was
# tuned on tens of thousands of real assistant turns; it is benchmark-agnostic.)
#
# IMPORTANT — natural-language PREAMBLE before <tool_call> is ALLOWED (the fix
# that mattered): the model's dominant correct shape is a short sentence ("Let me
# read the config file.") FOLLOWED by the call. The old _FORMAT_TOOLCALL_RE pinned
# <tool_call> to the very start (after optional <think>), so every "preamble +
# call" turn fell through to the respond branch, hit <tool_call> in the tail, and
# was scored format-bad. Allowing the preamble flips trajectory format_ok up with
# zero new false-OKs.
#
# We still reject genuine structural garbage: more than one <tool_call>, an
# orphan/unclosed </tool_call>, text AFTER </tool_call> (the rollout STOPs on
# </tool_call>, so a real tool turn ends exactly there), leftover
# <tool_response>/<|im_*|> markup, or a malformed/double <think>.
# A tool turn's PREAMBLE is also scanned for that markup (only the user-visible
# reply text before the call may be free-form). Bad turns pay an additive
# FORMAT_BAD_PENALTY on task reward.
#
# _FORMAT_TOOLCALL_RE: optional <think>, then arbitrary preamble (captured so the
# caller can scan it for markup), then EXACTLY ONE <tool_call>...</tool_call> with
# nothing but whitespace after it. The inner `(?:(?!</tool_call>).)*` is a
# tempered-dot so the block can't swallow a second close tag.
# The think prefix is OPTIONAL and comes in TWO shapes:
#   * no-think / closed:   <think>...</think>   (opening tag present)
#   * think-ON / ORPHAN:   ...</think>          (NO opening tag — Qwen3.5 puts the
#         opening <think> in the generation PROMPT, so the model's sampled turn
#         starts with raw reasoning and only emits the CLOSING </think>.)
# `_THINK_PREFIX` optionally consumes one think block: an OPTIONAL opening
# `<think>`, then a reasoning body that contains NO further `<think>`/`</think>`
# (tempered dot `(?:(?!</?think>).)*` — forbids nesting / a second block), then
# the closing `</think>`. The whole thing is optional (plain turns have no think).
_THINK_PREFIX = r"(?:(?:<think>)?(?:(?!</?think>).)*</think>\s*)?"
_FORMAT_TOOLCALL_RE = re.compile(
    r"\A\s*" + _THINK_PREFIX +
    r"(?P<pre>(?:(?!</?think>).)*?)<tool_call>(?!.*<tool_call>)(?:(?!</tool_call>).)*</tool_call>\s*\Z",
    re.DOTALL,
)
_FORMAT_THINK_PREFIX_RE = re.compile(
    r"\A\s*" + _THINK_PREFIX + r"(?P<after>.*)\Z",
    re.DOTALL,
)
# Markup that must never appear in free-form text (a respond turn's whole body, or
# a tool turn's preamble). Shared by both branches of _assistant_turn_format_ok.
_FORMAT_MARKUP_GARBAGE = (
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
)
# Additive penalties subtracted from total_reward when any assistant turn is
# format-bad. Fires on genuine structural garbage instead of on every correct tool
# turn. Weakened so format noise can't dominate the task signal:
#   - Successful (raw>0): pay 0.1 — light tap, still leaves +0.9 reward.
#   - Failed (raw==0): no penalty. We rely on the task gradient (and SFT) for
#     format learning; double-charging failures was hurting more than helping.
FORMAT_BAD_PENALTY_SUCCESS = 0.1
FORMAT_BAD_PENALTY_FAIL = 0.0

# Think length penalty is DISABLED. Earlier ablation showed it created a
# perverse gradient ("failed + long think" got -0.3, worse than truncation),
# pulling the model away from the SFT init's long-CoT distribution. We still
# measure avg_think_chars below for wandb monitoring, but the penalty is forced 0.
THINK_BUDGET_PER_TURN_CHARS = 500
THINK_PENALTY_PER_OVERAGE_CHAR = 0.0
THINK_PENALTY_CAP = 0.0
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_think_for_user(text: str) -> str:
    """Strip the reasoning (everything up to and including the LAST </think>) from
    a RESPOND turn's text BEFORE it is handed to the environment.

    WHY (think-ON only): with enable_thinking, the model's output is
    `reasoning...</think>\\n\\nvisible reply` (the opening <think> lives in the
    prompt, so the sampled text is an ORPHAN-close block). The tool parser's
    `normal_text` does NOT remove this — only tool-call markup. Workspace-Bench has
    no user simulator, but we still strip so any text the env echoes back / logs is
    the agent's visible reply, not its internal monologue. no-think turns have no
    </think> so this is a no-op.
    Note: we only strip the copy SENT to the env; the trained `cur_response`
    (loss-mask source) is left untouched."""
    if "</think>" not in text:
        return text
    return text.rsplit("</think>", 1)[1].lstrip("\n")


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
    "YOU MUST EXECUTE TOOLS (write_file / edit_file) TO MAKE ANY CHANGE TO THE "
    "WORKSPACE — describing a change in text is NOT enough.\n"
    "Explore with read_file / list_dir / grep before editing. Each tool call "
    "leads to a message returned by the system.\n"
    "When the task is fully complete and all required output files exist, call "
    "the `finish` function.\n"
)

# Appended to TOOL_INSTRUCTION ONLY when enable_thinking is on (think-ON). The
# Qwen3.5 template opens a `<think>` in the generation prompt, but NOTHING in the
# system prompt tells the model the reasoning contract — so it free-runs (long,
# unbounded CoT) and can confuse the `<think>` REASONING TAG with the `think`
# TOOL (a real function called via <tool_call><function=think>). This clause
# defines the contract and disambiguates the two. Kept short to bound CoT length.
THINK_INSTRUCTION = (
    "\nReasoning format: BEGIN every turn by opening a <think> block, write your "
    "private step-by-step reasoning, then close it with </think>. After "
    "</think>, output EITHER a single <tool_call>...</tool_call> OR a plain-text "
    "reply — never both. Keep the reasoning concise.\n"
    "Note: the <think>...</think> reasoning tag is NOT the `think` function. To "
    "record a durable thought as a tool, call <tool_call><function=think>...; to "
    "merely reason before acting, just use the <think> block.\n"
)


class TrainableAgentMixin:
    """
    Mixin class that provides trainable agent functionality for Workspace-Bench.

    This mixin extends the tool-calling agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Inject the file-agent tool-use guidance into the rendered chat-template
        prompt (one-or-no call per turn / MUST execute tools to edit / call finish).

        IMPORTANT — template-format dependent:
          * Qwen3 / Hermes templates carry the sentence
                "You may call one or more functions to assist with the user query."
            which we replace outright (we allow at most ONE call per turn).
          * Qwen3.5-4B-Base / Qwen3-Coder use the XML tool format and DO NOT
            contain that sentence at all — its tools block reads
                "If you choose to call a function ONLY reply in the following
                 format with NO suffix: <tool_call><function=...>..."
            so a naive `.replace(...)` would be a NO-OP there and the guidance
            would be silently dropped. For that template we insert TOOL_INSTRUCTION
            just before the format spec instead.

        Applied identically at sampling and training render time (both go through
        _render_messages_text), so the prompt prefix stays byte-consistent and
        the assistant-boundary alignment in _build_training_tensor is unaffected.
        """
        instruction = TOOL_INSTRUCTION
        if self._enable_thinking:
            instruction = instruction.rstrip("\n") + "\n" + THINK_INSTRUCTION

        # Idempotency guard: render output is always fresh template text (never
        # contains TOOL_INSTRUCTION), so this normally no-ops; the guard just
        # prevents a double-injection if this is ever called on already-patched
        # text (the qwen3.5 branch keeps the anchor line, so a naive re-run would
        # inject twice and desync the sampling/training prefixes).
        if TOOL_INSTRUCTION.strip() in text:
            return text

        qwen3_anchor = "You may call one or more functions to assist with the user query."
        if qwen3_anchor in text:
            return text.replace(qwen3_anchor, instruction)

        # Qwen3.5 / qwen3-coder XML template: inject before the format spec line.
        qwen35_anchor = "If you choose to call a function ONLY reply in the following format"
        if qwen35_anchor in text:
            return text.replace(qwen35_anchor, instruction.strip() + "\n\n" + qwen35_anchor, 1)

        # Neither anchor present (template drift): warn ONCE, leave prompt intact
        # rather than crash. Tool parsing can still work; only the behavioral
        # guidance is absent, so this is a soft degradation, not a hard failure.
        if not getattr(self, "_warned_no_tool_anchor", False):
            logger.warning(
                "Workspace-Bench TOOL_INSTRUCTION anchor not found in chat template; "
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
            env: Workspace-Bench environment instance
            action: Action to execute

        Returns:
            Environment step result
        """
        return await env.step(action)

    async def _initialize_environment(self, env, task_index: int | None) -> tuple[str, dict[str, Any]]:
        """
        Initialize the environment and get initial observation.

        Args:
            env: Workspace-Bench environment instance
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

    @property
    def _enable_thinking(self) -> bool:
        """Hyperparameter: whether the chat template runs in thinking mode.

        Read from env WS_ENABLE_THINKING (default True = think-ON; set 0 for the
        no-think path). The earlier think-ON crash was the FORMAT regex requiring
        an opening <think> tag (fixed in _THINK_PREFIX), not token alignment.
        Set by the run script from a CLI/env knob so it is consistent across all
        rollout actors (the value is injected into the Ray job runtime-env, like
        the other WS_* vars). Cached on first access.
        """
        cached = getattr(self, "_enable_thinking_cached", None)
        if cached is None:
            cached = os.environ.get("WS_ENABLE_THINKING", "1").lower() in ("1", "true", "yes")
            self._enable_thinking_cached = cached
        return cached

    @property
    def _strip_historical_think(self) -> bool:
        """Hyperparameter: in think-ON mode, drop the `<think>...</think>` block
        from every assistant turn EXCEPT the one currently being generated/trained.

        WHY (the bug this fixes): the Qwen3.5 chat template only strips a historical
        assistant turn's `<think>` once a LATER real `user` turn advances
        `last_query_index`. But the dominant trajectory shape is a single user
        request followed by a long `tool` chain, and `tool` results render as
        `<|im_start|>user\\n<tool_response>...` which the template EXPLICITLY skips
        when advancing `last_query_index`. So `last_query_index` never moves, NO
        historical `<think>` is ever stripped, and every turn's full CoT is carried
        in the trained tokens AND re-fed as context on every subsequent turn —
        context grows linearly in the number of tool calls.

        With this ON (default in think-ON mode) we pre-strip the reasoning from all
        but the last assistant turn ourselves BEFORE apply_chat_template, on BOTH
        the sampling render (ag=True) and the training render (ag=False). The
        template then re-injects an empty `<think>\\n\\n</think>\\n\\n` wrapper for
        the stripped turns, which `_render_messages_text` collapses away.

        Read once from env WS_STRIP_HISTORICAL_THINK (default "1"). Set to 0 to
        keep the legacy behavior (full CoT in every historical turn). No-op when
        think is OFF. Cached on first access.
        """
        cached = getattr(self, "_strip_hist_think_cached", None)
        if cached is None:
            cached = (
                self._enable_thinking
                and os.environ.get("WS_STRIP_HISTORICAL_THINK", "1").lower() in ("1", "true", "yes")
            )
            self._strip_hist_think_cached = cached
        return cached

    @staticmethod
    def _strip_think_block(content: str) -> str:
        """Remove a leading reasoning block (everything up to and incl. the LAST
        `</think>`) from an assistant turn's content, leaving only the visible
        action/reply. Mirrors `_strip_think_for_user` but for the trained/rendered
        history. No-op when there is no `</think>`."""
        if not content or "</think>" not in content:
            return content
        return content.rsplit("</think>", 1)[1].lstrip("\n")

    def _render_messages_text(
        self,
        state: GenerateState,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        """Render messages via apply_chat_template + tool-instruction patch.

        enable_thinking=False is CRITICAL for Qwen3.5 (and any Qwen "thinking"
        model). Its chat template, with add_generation_prompt=True and thinking
        ON (the default), appends a LONE opening `<think>\\n` to the prompt — so
        the opening tag lives in the PROMPT, the model's sampled response is
        `reasoning…</think>…` (an ORPHAN close), and _build_training_tensor's
        think-preserved / think-stripped alignment matches neither -> ~2/3 of
        response tokens get loss_mask=0 and format_ok==0 on every turn, which
        distorts the gradient. With enable_thinking=False the template instead
        emits a CLOSED `<think>\\n\\n</think>\\n\\n` and the agent runs no-think.
        Passed on BOTH the sampling render (add_generation_prompt=True) and the
        training render (add_generation_prompt=False) since both go through here,
        keeping the per-token prefix byte-consistent.

        Some HF tokenizers don't accept enable_thinking; pass it via a try/except
        so this stays safe on non-Qwen templates.
        """
        # enable_thinking is a HYPERPARAMETER (env WS_ENABLE_THINKING). When False
        # the template emits a CLOSED empty `<think>\n\n</think>\n\n` and the agent
        # runs no-think. When True the template emits a LONE opening `<think>\n`
        # into the generation prompt and the model samples `reasoning…</think>…`;
        # the sampled-vs-render token streams then differ per turn (see the case-(b)
        # alignment in _build_training_tensor). When think is ON,
        # _build_training_tensor surfaces a per-trajectory align-coverage metric to
        # wandb so a broken alignment shows up in the FIRST few steps. MUST be read
        # identically at sampling (ag=True) and training (ag=False) render so the
        # per-token prefix stays byte-consistent.
        # HISTORICAL-THINK STRIP (default in think-ON mode; see
        # _strip_historical_think). Pre-strip the `<think>...</think>` reasoning
        # from every assistant turn EXCEPT the one being generated/trained, so the
        # rendered context carries CoT only on the current turn. We must do this
        # ourselves because the template only strips a historical turn's think once
        # a LATER real `user` turn advances last_query_index — and tool-response
        # turns (rendered as <tool_response> "user" turns) are explicitly skipped
        # by that logic, so in a pure tool chain NO historical think is ever dropped
        # by the template alone.
        #   * Sampling render (ag=True): NONE of `messages` is the current turn yet
        #     (the current turn is what we're about to sample, appended only after),
        #     so strip every assistant turn. The prompt then ends at
        #     `<|im_start|>assistant\n` (after the THINK-SELF-OPEN trim below) and
        #     the model opens its own fresh <think>.
        #   * Training render (ag=False): keep the LAST assistant turn's think (that
        #     is the turn whose CoT we train), strip all earlier ones. Stripped
        #     turns render content-only, which _build_training_tensor already aligns
        #     via its think-stripped case (b).
        # Stripped turns are NOT mutated in place — we render from a shallow-copied
        # message list so `messages` (and the trained `cur_response`) keep full CoT.
        render_messages = messages
        if self._strip_historical_think:
            keep_idx = -1
            if add_generation_prompt is False:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "assistant":
                        keep_idx = i
                        break
            render_messages = [
                ({**m, "content": self._strip_think_block(m.get("content", "") or "")}
                 if (m.get("role") == "assistant" and i != keep_idx) else m)
                for i, m in enumerate(messages)
            ]

        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=self.tools_info,
            enable_thinking=self._enable_thinking,
        )
        try:
            text = state.tokenizer.apply_chat_template(render_messages, **kwargs)
        except TypeError:
            # Template/tokenizer doesn't support enable_thinking — fall back.
            kwargs.pop("enable_thinking", None)
            text = state.tokenizer.apply_chat_template(render_messages, **kwargs)
        text = self._reformulate_tool_call(text)

        # Collapse the empty `<think>\n\n</think>\n\n` wrappers the template
        # re-injects for the assistant turns we just stripped (its non-reasoning
        # assistant arm emits a closed empty block when reasoning_content is empty).
        # Without this, every stripped historical turn would carry a hollow think
        # block — pure token bloat and a magnet for the format check. Gated on
        # _strip_historical_think so it ONLY runs in the think-ON strip path: in
        # no-think mode that exact substring is the legitimate generation-prompt
        # block (handled separately below) and must NOT be removed.
        if self._strip_historical_think:
            text = text.replace("<think>\n\n</think>\n\n", "")

        # THINK-SELF-OPEN (default in think-ON mode). Qwen3.5's template, with
        # add_generation_prompt=True and thinking ON, appends a LONE opening
        # `<think>\n` to the generation prompt. That made the OPENING tag live in
        # the PROMPT, so the model only ever sampled an ORPHAN `reasoning…</think>…`
        # — which _build_training_tensor had to re-align via case (b'), and which
        # was the root cause of the reward decline (the sampled stream never
        # carried the `<think>` token, alignment drifted, ~2/3 of tokens went
        # untrained). We strip that injected `<think>\n` ONLY on the SAMPLING
        # render (add_generation_prompt=True) so the prompt ends at
        # `<|im_start|>assistant\n` and the model samples its OWN complete
        # `<think>\n…\n</think>\n\n…` block. The opening tag is now a trained token,
        # and the last assistant turn's canonical render (ag=False) matches the
        # sample exactly -> _build_training_tensor case (a), not the lossy (b').
        #   * Gated on self._enable_thinking: in no-think mode the suffix is the
        #     CLOSED `<think>\n\n</think>\n\n` block, which we must NOT strip.
        #   * Gated on add_generation_prompt: the training render (ag=False) has
        #     no generation prompt / no trailing `<think>`, so it is untouched.
        #   * The endswith() guard makes this a safe no-op on any template that
        #     does not inject a lone opening `<think>\n`.
        if add_generation_prompt and self._enable_thinking and text.endswith("<think>\n"):
            text = text[: -len("<think>\n")]
        return text

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
             This is what slime backprops on. In think-ON mode _render_messages_text
             strips historical <think> from all but the LAST assistant turn (see
             _strip_historical_think), so only the final turn carries CoT; earlier
             turns render content-only. (We drive this ourselves because the
             template's last_query_index logic does NOT strip across tool-response
             turns.)
          2. Locate every `<|im_start|>assistant\\n` boundary in the rendered tokens
             and pair the i-th boundary with turn_samples[i].
          3. For each pair, the rendered span is one of:
               a) full sample (think preserved — the LAST assistant turn, whose
                  think we keep): span == sampled_token_ids → mask=1, logprobs as-is
               b) post-think suffix (think stripped — every earlier turn): span ==
                  sampled_token_ids[k:] where k counts <think>...</think>\\n\\n
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
        n_align_fail = 0
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
            n_align_fail += 1
            logger.warning(
                f"Turn {span_idx}: render/sample alignment failed "
                f"(span_len={len(span_tokens)}, sampled_len={len(sampled_tokens)}); "
                f"no training signal for this turn."
            )

        # Alignment-coverage guardrail: how many assistant spans failed to align
        # (case c) out of those we attempted. Surfaced to wandb by asolve so a
        # broken think-on alignment (the historical "2/3 tokens untrained" bug)
        # is visible in the FIRST steps rather than hours later via reward decay.
        self._last_align_fail_turns = n_align_fail
        self._last_align_total_turns = n_pairs

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
        wrap). In think-ON mode we also strip historical `<think>` ourselves (see
        `_render_messages_text` / `_strip_historical_think`): the template's own
        last_query_index logic only drops a turn's think once a LATER real user
        turn arrives, but tool-response turns do NOT advance last_query_index, so a
        pure tool chain would otherwise carry every turn's full CoT forever. The
        training tensor is rebuilt at end-of-trajectory from the same final render,
        with per-turn sampled logprobs aligned back onto the surviving tokens —
        see `_build_training_tensor` for the alignment cases.

        The episode ends when the agent calls the `finish` terminate tool (the env
        sets done=True and computes the rubric-judge reward), or when it hits
        max_num_steps / a length-truncation.
        """
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"

        # Session-affinity routing for this multi-turn trajectory. When the router
        # policy is consistent_hashing, every turn of THIS trajectory must carry
        # the SAME routing key so the router hashes them all to one worker, which
        # then reuses its prefix cache across turns (the bulk of a trajectory's
        # prompt is the unchanged conversation history). The key is generated ONCE
        # here — not per turn — so it is stable for the whole `for` loop below.
        # We drive the actor via this custom path (not slime's sglang_rollout),
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
        # language message, no <tool_call>) burns through max_new_tokens after
        # emitting its own <|im_end|>.
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

        # ── Rollout time-attribution (async overlap diagnostic) ──
        # Within a single trajectory the wall-clock between env.reset and the
        # terminal turn splits across three buckets we separate on wandb:
        #   t_actor — waiting on the actor's own SGLang generation (_call_llm).
        #   t_judge — waiting on env.step() for the terminal `finish` action, which
        #             runs the BLOCKING Bedrock rubric judge (the reward). This is
        #             the "waiting for the reward model" cost.
        #   t_env   — env.reset() (workspace materialization) + every non-terminal
        #             env.step() (local file-tool invoke). Cheap disk I/O, measured
        #             so it isn't charged to the judge bucket.
        # Times use a monotonic clock; under asyncio.gather they overlap across
        # trajectories — so these are PER-TRAJECTORY sums, not a partition of the
        # batch wall-clock. The ratio t_judge/(t_actor+t_judge+t_env) is the
        # quantity of interest: what fraction of a trajectory's blocking time is
        # spent waiting on the judge vs. the agent's own generation.
        t_actor = 0.0
        t_judge = 0.0
        t_env = 0.0

        # env.reset materializes the workspace overlay (disk I/O). Guard it so one
        # flaky sample doesn't take down the whole asyncio.gather.
        try:
            _t0 = time.monotonic()
            obs, info = await self._initialize_environment(env, task_index)
            t_env += time.monotonic() - _t0
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
            # Re-render the current message list each turn. In think-ON mode
            # _render_messages_text strips `<think>` from every historical
            # assistant turn, so the context grows sub-linearly in the number of
            # tool calls instead of carrying every turn's full CoT.
            input_text = self._render_messages_text(state, messages, add_generation_prompt=True)

            payload = {
                "text": input_text,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            try:
                _t0 = time.monotonic()
                output = await self._call_llm(url, payload, headers=routing_headers)
                t_actor += time.monotonic() - _t0
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
            # of handing a half-written message to the environment. The truncated
            # turn's training signal is later dropped by _build_training_tensor
            # (its render is unreliable when </think> never closed).
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
            # think-ON: the parser's normal_text still carries the reasoning +
            # </think>. Strip it so the env / logs see ONLY the agent's visible
            # reply, not its internal monologue. No-op for no-think turns. Only
            # affects the copy sent to the env; the trained cur_response (recorded
            # above) keeps the full text for loss-mask.
            agent_content = _strip_think_for_user(agent_content)
            action = call_to_action_sglang(calls, agent_content)

            # The terminal `finish` action's env.step runs the BLOCKING Bedrock
            # rubric judge (the reward); every other action is a local file-tool
            # invoke. Attribute the env.step wall-time to the matching bucket.
            is_judge_step = action.name == FINISH_TOOL_NAME
            try:
                _t0 = time.monotonic()
                env_response = await self._execute_tool(env, action)
                _dt = time.monotonic() - _t0
                if is_judge_step:
                    t_judge += _dt
                else:
                    t_env += _dt
            except Exception as e:
                logger.warning("Environment step failed (typically a judge call error).")
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

        # Direct CoT-length penalty: DISABLED (cap=0). We still measure
        # avg_think_chars for wandb monitoring.
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
        # Alignment-coverage guardrail (esp. for think-on). frac_trained is the
        # fraction of response tokens that got a training signal; align_fail_turns
        # is how many assistant spans fell to case (c). If think-on re-breaks the
        # sampled-vs-render alignment, frac_trained collapses (~0.3) and
        # align_fail_turns spikes in the FIRST steps — watch these on wandb.
        info["frac_trained"] = float(sum(loss_masks) / len(loss_masks)) if loss_masks else 0.0
        info["align_fail_turns"] = int(getattr(self, "_last_align_fail_turns", 0))
        info["enable_thinking"] = int(self._enable_thinking)

        # ── Rollout time-attribution: actor self-rollout vs. waiting on the rubric
        # judge (see the t_* accumulators above). Surfaced to wandb by
        # compute_metrics_from_samples; the headline number is
        # rollout/traj_judge_time_frac/mean — the fraction of a trajectory's
        # blocking time spent waiting for the Bedrock judge to score the rubrics.
        traj_blocking_time = t_actor + t_judge + t_env
        info["traj_actor_time"] = float(t_actor)
        info["traj_judge_time"] = float(t_judge)
        info["traj_env_time"] = float(t_env)
        info["traj_blocking_time"] = float(traj_blocking_time)
        info["traj_judge_time_frac"] = (
            float(t_judge / traj_blocking_time) if traj_blocking_time > 0 else 0.0
        )
        info["traj_actor_time_frac"] = (
            float(t_actor / traj_blocking_time) if traj_blocking_time > 0 else 0.0
        )

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

    This agent combines the ToolCallingAgent shell with the TrainableAgentMixin to
    support async interaction with sglang servers for reinforcement learning.
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
