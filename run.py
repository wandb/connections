import asyncio
import time
import json
import random
from dataclasses import dataclass

import weave
import openai

import simple_parsing

client = openai.AsyncClient()

SLEEP_TIME = 0.2

@dataclass
class ScriptArgs:
    model: str = "gpt-4o"
    weave_project: str = "connections_refactor"
    file_path: str = "connections_prompts.jsonl"
    max_retries: int = 4
    max_tokens: int = 128
    temperature: float = 0.7
    num_samples: int = 5

args = simple_parsing.parse(ScriptArgs)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


@weave.op()
async def call_openai(messages, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={ "type": "json_object" }
        )
    extracted = response.choices[0].message.content
    if extracted is None:
        raise ValueError("No response from model")
    return extracted

@weave.op()
async def generate_solution(messages):

    res = await call_openai(messages)
    try:
        generation = json.loads(res)
    except:
        generation = {}
    return generation

@weave.op()
def check_one_solution(solution, model_output):
    gen_reason= model_output["reason"]
    gen_words = model_output["words"]
    for sol_dict in solution:
        sol_words = sol_dict["words"]
        sol_reason = sol_dict["reason"]
        if set(gen_words) == set(sol_words):
            print(f"{gen_reason} ~ {sol_reason}: {gen_words} == {sol_words}")
            return {"match": 4}
        elif len(set(gen_words).intersection(set(sol_words))) == 3:
            return {"match": 3}
    else: 
        return {"match": 0} 


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
    "Provide the one group you are most sure of as your final answer. I will enter this into the puzzle and give you feedback. I will tell you whether it is correct, incorrect or if you got 3 out of 4. "
    "Then we will continue until the puzzle is solved, or you lose.\n"
    """The results should be in JSON format as following: {"reason":"reason why words are grouped", "words":["word1", "word2", "word3", "word4"]}\n"""
)


class Model(weave.Model):
    system_prompt: str
    user_prompt: str
    max_retries: int = 4
    shuffle_words: bool = False

    @weave.op()
    def create_incorrect_prompt(self, words: list[str], solution: dict, score: str):
        incorrect_prompt = f"This solution is wrong: {solution} " + (f"but you got 3/4 words that belong to a category. Try changing one word to get a 4/4." if score["match"] == 3 else "")
        incorrect_prompt += (
            "\nDon't repeat any previous wrong solutions. Let's try again. "
            "Let's continue with the rest of the words. "
            f"Here are the remaining {len(words)} words: {words} \n"
            "Guess another group of 4 words. Think outside the box and don't repeat the same previous wrong answer please. "
            "Do not add any additional text to your final answer, just the category name and the 4 words."
        )
        return incorrect_prompt

    @weave.op()
    def create_correct_prompt(self, words: list[str], solution: dict):
        if self.shuffle_words:
            random.shuffle(words)
        correct_prompt = (
            f"Great work, {solution} is a correct solution. "
            "Let's continue with the rest of the words. "
            f"Here are the remaining {len(words)} words:\n{words} \n"
            "Guess another group of 4 words. "
            "Do not add any additional text to your final answer, just the category name and the 4 words."
        )
        return correct_prompt
    
    @weave.op()
    def initial_messages(self, words):
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_prompt + f"Here are the starting 16 words: {words}\nDo not add any additional text to your final answer, just the group name and the 4 words."
            }
        ]

    @weave.op()
    async def predict(self, words, solution):
        retries = 0
        correct_guesses = []
        remaining_words = [w for w in words]  # listify

        # initial prompt
        messages = self.initial_messages(words)

        while len(remaining_words) > 4 and retries<self.max_retries:
            # generate a solution for the current group
            generation = await generate_solution(messages)
            time.sleep(SLEEP_TIME)
            scores = check_one_solution(solution, generation)
            print(f"Current generation {generation} -> score: {scores}")
            time.sleep(SLEEP_TIME)
            if scores["match"] == 4:
                print(" > Great, we have a match")
                correct_guesses.append(generation)
                remaining_words = [w for w in remaining_words if w not in generation["words"]]
                user_prompt = self.create_correct_prompt(remaining_words, generation)
            else:
                print(f" > Not a match, let's try again: retries={retries}")
                user_prompt = self.create_incorrect_prompt(remaining_words, generation, scores)
                retries+=1
            # we append to the messages list
            messages += [
                {
                    "role": "assistant",
                    "content": str(generation)
                },
                {
                    "role": "user",
                    "content": str(user_prompt)
                }
            ]
        # we have the last group in here!
        if len(remaining_words) == 4: 
            print("We have the last group in here!")
            correct_guesses.append({"reason": "last_group", "words": remaining_words})
        return correct_guesses


@weave.op()
def check_final_solution(solution, model_output):
    "Check that all group of words match the solution"
    solution_set = {frozenset(group["words"]) for group in solution}
    model_output_set = {frozenset(group["words"]) for group in model_output}
    
    accuracy = len(solution_set.intersection(model_output_set))
    
    return {"match": accuracy == 4, "accuracy": accuracy}


weave.init(args.weave_project)

model = Model(system_prompt=system_prompt, user_prompt=user_prompt, max_retries=args.max_retries)

# ds = load_jsonl('connections_prompts2.jsonl')
ds = load_jsonl(args.file_path)


# print(asyncio.run(model.predict(ds[0]["words"], ds[0]["solution"])))

weave_eval = weave.Evaluation(dataset=ds[:args.num_samples], scorers=[check_final_solution])
print(asyncio.run(weave_eval.evaluate(model)))

