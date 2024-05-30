import asyncio
import json
import weave
import openai
from dataclasses import dataclass
import simple_parsing

client = openai.AsyncClient()

@dataclass
class ScriptArgs:
    project: str = "connections_demo"
    file_path: str = "connections_prompts.jsonl"
    temperature: float = 0.7
    max_tokens: int = 256
    num_samples: int = 5

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

@weave.op()
async def call_openai(messages, model="gpt-4o", max_tokens=256, temperature=0.7):
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
async def generate_solution(messages, **kwargs):

    res = await call_openai(messages, **kwargs)
    try:
        generation = json.loads(res)
    except:
        generation = {}
    return generation

class OneShotModel(weave.Model):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    
    @weave.op()
    async def predict(self, words):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt + str(list(words))}
        ]
        return await generate_solution(messages, temperature=self.temperature, max_tokens=self.max_tokens)

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
    """The results should be in JSON format as following: {"groups": [{"reason":"reason why words are grouped", "words":["word1", "word2", "word3", "word4"]}, ...]}"""
    "Provide a full solution to the puzzle, it should be 4 groups of 4 words."
    "Here are the words for today’s puzzle:\n")



@weave.op()
def check_final_solution(solution, model_output):
    "Check that all group of words match the solution"
    solution_set = {frozenset(group["words"]) for group in solution["groups"]}
    model_output_set = {frozenset(group["words"]) for group in model_output["groups"]}
    
    accuracy = len(solution_set.intersection(model_output_set))
    
    return {"match": accuracy == 4, "accuracy": accuracy}


if __name__ == "__main__":
    args = simple_parsing.parse(ScriptArgs)

    weave.init(args.project)

    model = OneShotModel(system_prompt=system_prompt, user_prompt=user_prompt, temperature=args.temperature, max_tokens=args.max_tokens)

    ds = load_jsonl(args.file_path)

    weave_eval = weave.Evaluation(dataset=ds[-args.num_samples:], scorers=[check_final_solution])
    print(asyncio.run(weave_eval.evaluate(model)))
