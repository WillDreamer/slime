from inspect_ai import Task, eval as ieval, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, GenerateConfig


def desc(m):
    try:
        base = getattr(getattr(m, "api", None), "base_url", "?")
        return f"{m.name} | base={base}"
    except Exception as e:
        return f"{m} ({e})"


@solver
def probe():
    async def solve(state: TaskState, generate: Generate):
        a = get_model(config=GenerateConfig(temperature=0.0))  # exactly as agents.py
        try:
            u = get_model(role="user", config=GenerateConfig(temperature=0.0))
            ud = desc(u)
        except Exception as e:
            ud = f"ERR {e}"
        print("AGENT(default):", desc(a))
        print("USER(role=user):", ud)
        return state
    return solve


@task
def t():
    return Task(dataset=[Sample(input="hi")], solver=probe())


agent = get_model("openai/Qwen3-8B-Base-Math", base_url="http://127.0.0.1:7001/v1", api_key="dummy")
user = get_model("openai/GLM-4.7-Flash", base_url="http://127.0.0.1:7006/v1", api_key="dummy")

print("##### CASE A: model=OBJECT #####")
ieval(t(), model=agent, model_roles={"user": user}, display="none")
print("##### CASE B: model=STRING + model_base_url #####")
ieval(t(), model="openai/Qwen3-8B-Base-Math", model_base_url="http://127.0.0.1:7001/v1",
      model_roles={"user": user}, display="none")
