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


async def main():
    agent = get_model(
        "openai/Qwen3-8B-Base-Math",
        base_url="http://127.0.0.1:7001/v1",
        api_key="dummy",
        responses_api=False,
        config=GenerateConfig(temperature=0.0, max_tokens=512, max_retries=0, timeout=60),
    )
    msgs = [ChatMessageSystem(content="You are a retail agent. Use tools."),
            ChatMessageUser(content="What is the status of order 42?")]
    try:
        out = await agent.generate(msgs, tools=[make_tool()])
        print("OK:", (out.completion or "")[:150], flush=True)
    except Exception as e:
        # unwrap tenacity RetryError -> underlying exception
        inner = e
        la = getattr(e, "last_attempt", None)
        if la is not None:
            try:
                inner = la.exception() or e
            except Exception:
                pass
        if inner is e and getattr(e, "__cause__", None):
            inner = e.__cause__
        print("OUTER:", type(e).__name__, "| INNER:", type(inner).__name__, flush=True)
        for attr in ("message", "body", "status_code"):
            print(f"  {attr}:", repr(getattr(inner, attr, None))[:600], flush=True)
        resp = getattr(inner, "response", None)
        if resp is not None:
            try:
                print("  response.text:", resp.text[:1000], flush=True)
            except Exception:
                pass


asyncio.run(main())
