# Vendored Workspace-Bench environment for slime RL (AECE).
#
# Mirrors the structural role of examples/tau-bench/tau_bench so the (near-
# verbatim) trainable_agents.py / generate_with_ws.py / async_env.py keep
# importing the SAME symbol names (Action, RESPOND_ACTION_NAME, RunConfig,
# EnvResponse, EnvResetResponse, EnvInfo, RewardResult, get_env, Tool, Agent,
# ToolCallingAgent). The benchmark itself lives at
# https://github.com/OpenDataBox/Workspace-Bench — here we only re-implement the
# env/reward surface the RL rollout needs (file-editing tools + a Bedrock rubric
# judge), not the full upstream agent_runner / agent_as_a_judge harness.
