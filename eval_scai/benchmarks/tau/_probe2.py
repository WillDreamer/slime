import asyncio
import traceback

from inspect_ai.model import get_model, GenerateConfig, ChatMessageSystem, ChatMessageUser
from inspect_ai.tool import ToolDef


def make_tool():
    def get_order(order_id: int) -> str:
        "Get order by id."
        return "shipped"
    return ToolDef(get_order, name="get_order", description="Get order by id",
                   parameters={"order_id": "the order id"}).as_tool()


async def try_case(label, max_tokens):
    agent = get_model(
        "openai/Qwen3-8B-Base-Math",
        base_url="http://127.0.0.1:7001/v1",
        api_key="dummy",
        config=GenerateConfig(temperature=0.0, max_tokens=max_tokens,
                              max_retries=0, timeout=60),
    )
    print(f"\n##### {label}: max_tokens={agent.config.max_tokens} #####", flush=True)
    msgs = [ChatMessageSystem(content="You are a retail agent. Use tools."),
            ChatMessageUser(content="What is the status of order 42?")]
    try:
        new_messages, output = await agent.generate_loop(msgs, tools=[make_tool()])
        print("OK new_messages:", len(new_messages), "| completion:", (output.completion or "")[:150], flush=True)
    except Exception:
        print("=== TRACEBACK ===", flush=True)
        traceback.print_exc()


async def main():
    await try_case("A None max_tokens (agents.py default)", None)
    await try_case("B explicit max_tokens", 1024)


asyncio.run(main())
