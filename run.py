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
    file_path: str = "connections_prompts2.jsonl"
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
def check_one_solution(category_0, category_1, category_2, category_3, model_output):
    for sol_dict in [category_0, category_1, category_2, category_3]:
        sol_group = sol_dict["words"]
        for gen_cat, gen_group in model_output.items():
            if set(gen_group) == set(sol_group):
                print(f"{gen_cat} ~ {sol_dict['reason']}: {gen_group} == {sol_group}")
                return {"match": True}
            elif len(set(gen_group).intersection(set(sol_group))) == 3:
                return {"match": "3/4"}
    else: 
        return {"match": False} 
    #     return {"match": False} 


system_prompt_cot = (
    "You are an expert puzzle solver. You understand literature and you are well versed on word play. "
    "I want you to solve a daily word puzzle that finds commonalities between words.\n"
    )

user_prompt_cot = (
    "Here it's the puzzle:\n"
    "- There are 16 words, which form 4 groups of 4 words. Each group has some common theme that links the words.\n"
    "- You must use each of the 16 words, and use each word only once.\n"
    "- Each group of 4 words are linked together in some way. \n"
    "The connection between words can be simple.\n"
    """- An example of a simple connection would be {'types of fish': ["Bass", "Flounder", "Salmon", "Trout"]}. \n"""
    """- Categories can also be more complex, and require abstract or lateral thinking. An example of this type of connection would be {'things that start with FIRE': ['Ant', 'Drill', 'Island', 'Opal']}\n"""
    "Provide the one group you are most sure of as your final answer. I will enter this into the puzzle and give you feedback. I will tell you whether it is correct, incorrect or if you got 3 out of 4."
    "Then we will continue until the puzzle is solved, or you lose.\n"
    """The results should be in JSON format as following: {"name_of_category": ["word1", "word2", "word3", "word4"]}\n"""
)


class ModelCOT(weave.Model):
    system_prompt: str
    user_prompt: str
    max_retries: int = 4

    @weave.op()
    def create_incorrect_prompt(self, words: list[str], solution: dict, score: str):
        incorrect_prompt = f"This solution is wrong: {solution} " + (f"but you got {score} words that belong to a category. Try changing one word to get a 4/4." if score == "3/4" else "")
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
        correct_prompt = (
            f"Great work, {solution} is a correct solution. "
            "Let's continue with the rest of the words. "
            f"Here are the remaining {len(words)} words:\n{random.shuffle(words)} \n"
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
    async def predict(self, words, category_0, category_1, category_2, category_3):
        messages = self.initial_messages(words)
        retries = 0
        correct_guesses = []
        remaining_words = [w for w in words]
        user_prompt = self.user_prompt + f"Here are the starting 16 words: {words}\n. Do not add any additional text to your final answer, just the group name and the 4 words."

        while remaining_words and retries<self.max_retries:
            generation = await generate_solution(messages)
            time.sleep(SLEEP_TIME)
            print(f"Try {retries}. Remaining words: {remaining_words}, guess: {generation}")
            scores = check_one_solution(category_0, category_1, category_2, category_3, generation)
            time.sleep(SLEEP_TIME)
            if scores["match"]:
                print(" > Great, we have a match")
                correct_guesses.append(generation)
                remaining_words = [w for w in remaining_words if w not in next(iter(generation.values()))]
                user_prompt = self.create_correct_prompt(remaining_words, generation)
            else:
                user_prompt = self.create_incorrect_prompt(remaining_words, generation, scores)
                retries+=1
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
        return correct_guesses


@weave.op()
def check_final_solution(categories, model_output):
    "Check that all group of words match the solution"    
    try:
        accuracy = len(model_output)
    except:
        accuracy = 0
    return {"match": True if accuracy == 4 else False, "accuracy": accuracy/4}


weave.init(args.weave_project)

model = ModelCOT(system_prompt=system_prompt_cot, user_prompt=user_prompt_cot, max_retries=args.max_retries)

# ds = load_jsonl('connections_prompts2.jsonl')
ds = load_jsonl(args.file_path)


print(asyncio.run(model.predict(ds[0]["words"], ds[0]["category_0"], ds[0]["category_1"], ds[0]["category_2"], ds[0]["category_3"])))

# weave_eval = weave.Evaluation(dataset=ds[-args.num_samples:], scorers=[check_final_solution])
# print(asyncio.run(weave_eval.evaluate(model)))

