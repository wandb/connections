import asyncio
import time
import json
import random
from dataclasses import dataclass

import weave
import openai
import instructor
from pydantic import BaseModel, Field

import simple_parsing

client = instructor.from_openai(openai.AsyncClient())

SLEEP_TIME = 0.2

@dataclass
class ScriptArgs:
    model: str = "gpt-4o"
    weave_project: str = "connections_alpha"
    file_path: str = "connections_prompts.jsonl"
    max_tokens: int = 128
    temperature: float = 0.7
    num_samples: int = 5
    N: int = 3

args = simple_parsing.parse(ScriptArgs)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


class Reflection(BaseModel):
    analysis: str

    def __str__(self):
        return self.analysis

class Group(BaseModel):
    reason: str
    words: list[str] = Field(max_length=4, description="A list of 4 words")

    def __str__(self):
        return "{" + f"{self.reason}: {list(self.words)}" + "}"


class Solution(BaseModel):  
    opinion: str
    score: int
    groups: list[Group] = Field(max_length=4, description="A list of 4 groups of 4 words with their reasons")
    def __str__(self):
        return ("- [" + ",".join(str(group) for group in self.groups) + "]\n"
                + f"  Opinion: {self.opinion}\n"
                + f"  Score: {self.score}\n")

    def validate(self, puzzle: list[str]):
        total_words = []
        for g in self.groups:
            if not set(g.words).issubset(set(puzzle)):
                return False
            for w in g.words:
                if w not in total_words:
                    total_words.append(w)
        return len(total_words) == 16
    
    def __eq__(self, other: 'Solution'):
        solution_set = {frozenset(group.words) for group in self.groups}
        other_set = {frozenset(group.words) for group in other.groups}
    
        accuracy = len(solution_set.intersection(other_set))
    
        return {"match": accuracy == 4, "accuracy": accuracy}
    

class PossibleSolutions(BaseModel):
    solutions: list[Solution] = Field(max_length=100, description="A list of possible solutions to the puzzle")

    def __str__(self):
        return "\n".join(str(solution) for solution in self.solutions)



@weave.op()
async def call_openai(messages, response_model=Solution, model=args.model):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        )
    if response_model is None:
        return response.choices[0].message.content
    else:
        return response

@weave.op()
def solve(sample: dict) -> dict:
    system_prompt = (
        "You are an expert puzzle solver. You understand literature and you are well versed on word play. "
        "I want you to solve a daily word puzzle that finds commonalities between words.\n"
        )

    user_prompt = (
        "Here it's the puzzle:\n"
        "- There are 16 words, which form 4 groups of 4 words. Each group has some common theme that links the words.\n"
        "- You must use each of the 16 words, and use each word only once.\n"
        "- Each group of 4 words are linked together in some way. \n"
        "The connection between words can be simple.\n"
        """- An example of a simple connection would be {"reason":'types of fish', "words":["Bass", "Flounder", "Salmon", "Trout"]}. \n"""
        """- Categories can also be more complex, and require abstract or lateral thinking. An example of this type of connection would be {"reason": 'things that start with FIRE', "words": ['Ant', 'Drill', 'Island', 'Opal']}\n"""
        f"Here are the starting 16 words:\n{sample['words']}\n"
        "Give me an analysis of the puzzle and explore word relations. Does it lok hard? Are there multiple word connections?"

    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    reflections = asyncio.run(call_openai(messages, response_model=None))

    user_prompt = (
        f"Ok, taking the previous analysis, let's try generating {args.N} different solutions to this problem. This means {args.N} times 4 groups of 4 words each. Remember to use all words and to not repeat words across groups.\n"
        f"Score each solution from 1 to 7 and give me an opinion on why you picked that as a valid solution. Mark the solution as vadid if you think it's a good solution. If it's not a good solution, give me a reason why it's not a good solution."
    )

    messages += [
        {"role": "assistant", "content": str(reflections)},
        {"role": "user", "content": user_prompt}
    ]

    possible_solutions = asyncio.run(call_openai(messages, response_model=PossibleSolutions))


    user_prompt2 = f"""Great, now we have to check if the solutions are correct.
    - Analize the solution so the relation between words makes sense. 
    - Veryfy that the words are related to each other by the reason you provided.
    - Give a score from 1 to 7 to each solution.
    - Reflect on each solution and give me an honest opinion on how plausible each solution is to be the real solution.
    """

    validated_possible_solutions = PossibleSolutions(solutions=[s for s in possible_solutions.solutions if s.validate(sample["words"])])

    messages += [
        {"role": "assistant", "content": "Here are the possible solutions: \n\n" + str(validated_possible_solutions)},
        {"role": "user", "content": user_prompt2}
    ]

    analysis_of_solutions = asyncio.run(call_openai(messages, response_model=None))


    user_prompt3 = f"""Now that you have analized the solutions, I want you to give me the best solution. Argument why you picked this one.
    """

    messages += [
        {"role": "assistant", "content": analysis_of_solutions},
        {"role": "user", "content": user_prompt3}
    ]

    final_solution = asyncio.run(call_openai(messages, response_model=Solution))

    target_solution = Solution(puzzle=sample["words"],groups=sample["solution"], opinion="Real solution", score=7)

    @weave.op()
    def check_solution(solution, model_output):
        return solution == model_output

    return check_solution(final_solution, target_solution)

    # weave_eval = weave.Evaluation(dataset=ds[:args.num_samples], scorers=[check_final_solution])
    # print(asyncio.run(weave_eval.evaluate(model)))


weave.init(args.weave_project)

ds = load_jsonl(args.file_path)

sample = ds[-1]

solve(sample)
