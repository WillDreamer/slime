"""
Self-play variant of trainable_agents.py.

Replaces the external LLM user simulator (litellm/GLM API) with the same
SGLang rollout engine that serves the policy model.  RESPOND actions now
go through an async SGLang call instead of a blocking HTTP call to an
external service, so the rollout engine is no longer idle while waiting
for the user simulator.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, EnvInfo, EnvResponse, RunConfig
from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

STOP_SIGNAL = "###STOP###"


class ContextWindowManager:
    """
    Sliding window over multi-turn conversation history.

    Keeps the full message list for token tracking / loss mask computation,
    but exposes a truncated view (system prompt + last k turns) for SGLang
    inference, preventing context-length overflow in long episodes.

    A "turn" is one agent→env/user exchange, i.e. a pair of:
      (assistant message, tool/user response message)
    The system prompt and first user message (initial observation) are
    always included regardless of k.
    """

    def __init__(self, system_msg: dict, initial_user_msg: dict, k: int = 10):
        self.k = k
        # Fixed prefix: [system, initial_user]
        self.prefix: list[dict] = [system_msg, initial_user_msg]
        # All messages after the prefix, grouped as flat list
        # Each "turn" appends 2 messages: assistant + env/user response
        self.history: list[dict] = []

    def append(self, msg: dict) -> None:
        """Append a single message to history."""
        self.history.append(msg)

    def windowed_messages(self) -> list[dict]:
        """Return prefix + last k*2 messages (k agent turns + k env responses)."""
        cutoff = self.k * 2
        return self.prefix + self.history[-cutoff:] if len(self.history) > cutoff else self.prefix + self.history

    def all_messages(self) -> list[dict]:
        """Return the complete untruncated message list (for token tracking)."""
        return self.prefix + self.history


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


# ---------------------------------------------------------------------------
# Async SGLang-backed user simulator
# ---------------------------------------------------------------------------

class SGLangUserSimulationEnv:
    """
    Drop-in async replacement for LLMUserSimulationEnv.

    Uses the same SGLang router as the policy model so both the agent and
    the user simulator share the same inference engine, eliminating the
    external API round-trip.

    Conversation format (mirrors LLMUserSimulationEnv):
      - system: user-simulator system prompt
      - user:   agent message (the "other side" of the conversation)
      - assistant: user simulator reply
      - user:   next agent message
      - ...
    """

    def __init__(self, tokenizer: AutoTokenizer, url: str, sampling_params: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.url = url
        # Use a separate, calmer sampling config: shorter responses, lower temperature.
        self.sampling_params = {
            **sampling_params,
            "temperature": 0.7,
            "max_new_tokens": 256,
        }
        self.messages: list[dict[str, Any]] = []

    def _build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n") if instruction is not None else ""
        )
        return (
            f"You are a user interacting with an agent.{instruction_display}\n"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information "
            "that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction.\n"
            "- Only if the instruction goal is satisfied and THE EXECUTION IS CONFIRMED "
            f"generate '{STOP_SIGNAL}' as a standalone message without anything else.\n"
            "- Do not repeat the exact instruction. Use your own words.\n"
            "- Try to make the conversation as natural as possible."
        )

    async def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self._build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return await self._generate()

    async def step(self, agent_message: str) -> str:
        """Called each time the agent sends a message to the user."""
        self.messages.append({"role": "user", "content": agent_message})
        return await self._generate()

    async def _generate(self) -> str:
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            # No tools — user sim is plain chat
        )
        output = await post(self.url, {"text": text, "sampling_params": self.sampling_params})

        if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            logger.warning("SGLang aborted user-sim generation; returning STOP signal.")
            return STOP_SIGNAL

        response = output["text"]
        if response.endswith("<|im_end|>"):
            response = response[:-10]
        response = response.strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def get_total_cost(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Self-play trainable agent mixin
# ---------------------------------------------------------------------------

TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class SelfPlayAgentMixin:
    """
    Replaces TrainableAgentMixin's asolve() with a version that uses
    SGLangUserSimulationEnv instead of the external LLM user simulator.

    Only RESPOND actions hit the user sim (async SGLang call).
    Tool actions call env.step() directly — they are pure Python DB
    operations with no I/O, so they complete in microseconds.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        return text.replace(
            "You may call one or more functions to assist with the user query.",
            TOOL_INSTRUCTION,
        )

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await post(url, payload)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        return self.openai_adapter.parse_response_to_openai_format(response)

    def _get_token_delta(
        self, tokenizer: AutoTokenizer, messages: list[dict]
    ) -> tuple[list[int], list[int]]:
        curr = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=True, tokenize=False
            )
            new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
            return new_tokens, [1] * len(new_tokens)
        else:
            prev = tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=False, tokenize=False
            )
            new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
            return new_tokens, [0] * len(new_tokens)

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> InteractionResult:
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join(
            [msg.get("content", "") for msg in messages if msg["role"] == "assistant"]
        )
        res.response_length = len(loss_masks)
        return res

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
        context_window_k: int = 10,
    ) -> InteractionResult:
        state = GenerateState(rollout_args)
        agent_url = (
            f"http://{rollout_args.sglang_router_ip}:"
            f"{rollout_args.sglang_router_port}/generate"
        )

        # --- Reset env state without triggering the external user sim ---
        if task_index is not None:
            env.task_index = task_index
        env.data = env.data_load_func()
        env.task = env.tasks[env.task_index]
        env.actions = []

        # --- Create async SGLang user sim and get initial observation ---
        user_sim = SGLangUserSimulationEnv(
            tokenizer=state.tokenizer,
            url=agent_url,
            sampling_params=sampling_params,
        )
        initial_obs = await user_sim.reset(instruction=env.task.instruction)

        # Build initial agent conversation with sliding window manager
        ctx = ContextWindowManager(
            system_msg={"role": "system", "content": self.wiki},
            initial_user_msg={"role": "user", "content": initial_obs},
            k=context_window_k,
        )

        prompt_text = state.tokenizer.apply_chat_template(
            ctx.all_messages(), tokenize=False, add_generation_prompt=True, tools=self.tools_info
        )
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        loss_masks: list[int] = []
        response_token_ids: list[int] = []
        total_reward = 0.0
        info: dict[str, Any] = EnvInfo(task=env.task, source="user").model_dump()
        env_response: EnvResponse | None = None

        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})

        for _ in range(max_num_steps):
            # --- Policy model (agent) generates next action ---
            # Use windowed_messages() for inference to avoid context overflow.
            # all_messages() is used for token delta tracking (complete history).
            text_input = state.tokenizer.apply_chat_template(
                ctx.windowed_messages(), tokenize=False, add_generation_prompt=True, tools=self.tools_info
            )
            text_input = self._reformulate_tool_call(text_input)
            output = await self._call_llm(agent_url, {"text": text_input, "sampling_params": sampling_params})

            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
                )

            response = output["text"]
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            # Parse tool calls
            try:
                openai_result = self._parse_tool(response)
                if not openai_result["success"]:
                    logger.warning(f"Tool parse failed: {openai_result['error']}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
                    )
                parsed = openai_result["parsed_result"]
            except Exception as e:
                logger.warning(f"Tool parse exception: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
                )

            # Track agent tokens (loss_mask=1) using full history
            ctx.append({"role": "assistant", "content": response})
            agent_token_ids, agent_loss_mask = self._get_token_delta(state.tokenizer, ctx.all_messages())
            response_token_ids.extend(agent_token_ids)
            loss_masks.extend(agent_loss_mask)

            # Build action
            agent_content, calls = parsed["normal_text"], parsed["calls"]
            if calls:
                if len(calls) > 1:
                    logger.debug("Multiple tool calls; using first.")
                tool_call = calls[0]
                params = json.loads(tool_call["parameters"])
                action = Action(name=tool_call["name"], kwargs=params if isinstance(params, dict) else {})
            else:
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_content})

            # --- Execute action ---
            if action.name == RESPOND_ACTION_NAME:
                # Self-play: call SGLang user sim asynchronously
                try:
                    user_response = await user_sim.step(action.kwargs["content"])
                except Exception as e:
                    logger.warning(f"User sim step failed: {e}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
                    )

                done = STOP_SIGNAL in user_response
                reward = 0.0
                env_info = EnvInfo(task=env.task, source="user")

                if done:
                    reward_res = env.calculate_reward()
                    reward = reward_res.reward
                    env_info.reward_info = reward_res
                    env_info.user_cost = 0.0

                env_response = EnvResponse(
                    observation=user_response,
                    reward=reward,
                    done=done,
                    info=env_info,
                )
                env.actions.append(action)

                # Add user response to agent's message history (loss_mask=0)
                ctx.append({"role": "user", "content": user_response})
            else:
                # Tool call: pure Python DB operation, no I/O
                try:
                    env_response = env.step(action)
                except Exception as e:
                    logger.warning(f"Tool execution failed: {e}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
                    )

                ctx.append({
                    "role": "tool",
                    "name": action.name,
                    "content": env_response.observation,
                })

            # Track env/tool tokens (loss_mask=0) using full history
            env_token_ids, env_loss_mask = self._get_token_delta(state.tokenizer, ctx.all_messages())
            response_token_ids.extend(env_token_ids)
            loss_masks.extend(env_loss_mask)

            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if env_response.done:
                res.status = Status.COMPLETED
                break

        if env_response is None or not env_response.done:
            res.status = Status.TRUNCATED

        return self._build_final_result(
            res, total_reward, info, ctx.all_messages(), loss_masks, prompt_token_ids, response_token_ids
        )


# ---------------------------------------------------------------------------
# Concrete agent class
# ---------------------------------------------------------------------------

class TrainableToolCallingAgentSelfPlay(ToolCallingAgent, SelfPlayAgentMixin):
    """
    ToolCallingAgent with self-play user simulation via SGLang.

    Identical to TrainableToolCallingAgent except asolve() comes from
    SelfPlayAgentMixin, which routes user-simulator calls through the
    same SGLang engine instead of an external API.
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
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )
        self.rollout_args = rollout_args or {}
        self.sampling_params = sampling_params or {
            "temperature": 1.0,
            "max_new_tokens": 1024,
        }
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type="qwen25")


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
):
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgentSelfPlay(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
