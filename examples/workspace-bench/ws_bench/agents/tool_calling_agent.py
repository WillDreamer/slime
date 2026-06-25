"""ToolCallingAgent shell for Workspace-Bench.

This keeps ONLY the constructor that TrainableToolCallingAgent calls via
super().__init__ (tools_info / wiki / model / provider / temperature). The
training/rollout path drives inference through the in-cluster SGLang server
(see ../trainable_agents.py), so the litellm-based `.solve()` from upstream
tau-bench is intentionally NOT ported — importing litellm would make the package
unimportable in the Greenland container, and the training path never calls it.
"""

import json
from typing import Any, Dict, List, Optional

from ws_bench.agents.base import Agent
from ws_bench.envs.base import Env
from ws_bench.types import Action, RESPOND_ACTION_NAME, SolveResult


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        # The synchronous litellm eval loop is not used for RL training; the
        # trainable subclass overrides this with an async SGLang rollout (asolve).
        raise NotImplementedError(
            "ToolCallingAgent.solve is not implemented for Workspace-Bench RL; "
            "use TrainableToolCallingAgent.asolve (SGLang rollout) instead."
        )


def message_to_action(message: Dict[str, Any]) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
