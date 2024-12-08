import asyncio
import time
import json
import logging
import random
from dataclasses import dataclass

import weave
import openai

import simple_parsing

from agentic_tools import run_agentic_solver, run_agentic_simulator

client = openai.AsyncClient()

SLEEP_TIME = 0.2

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ScriptArgs:
    model: str = "gpt-4o"
    weave_project: str = "agentic_connections_solver"
    file_path: str = "connections_prompts.jsonl"
    max_retries: int = 4
    max_tokens: int = 128
    temperature: float = 0.7
    num_samples: int = 5


args = simple_parsing.parse(ScriptArgs)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


class AgenticModel(weave.Model):

    @weave.op()
    async def predict(self, words, solution):
        solver_output = await run_agentic_solver(words, solution)
        return solver_output


@weave.op()
def check_final_solution(solution, model_output):
    "Check that all group of words match the solution"
    solution_set = {frozenset(group["words"]) for group in solution["groups"]}
    model_output_set = {frozenset(group["words"]) for group in model_output}

    accuracy = len(solution_set.intersection(model_output_set))

    return {"match": accuracy == 4, "accuracy": accuracy}


weave.init(args.weave_project)

model = AgenticModel()

# ds = load_jsonl('connections_prompts2.jsonl')
ds = load_jsonl(args.file_path)


# print(asyncio.run(model.predict(ds[0]["words"], ds[0]["solution"])))

weave_eval = weave.Evaluation(
    dataset=ds[: args.num_samples], scorers=[check_final_solution]
)
print(asyncio.run(weave_eval.evaluate(model)))
