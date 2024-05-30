import os
import json
import asyncio
import httpx
import pathlib
import random
from dataclasses import dataclass
from tqdm import tqdm
import simple_parsing
from datetime import timedelta, datetime

random.seed(42)
URL = "https://www.nytimes.com/svc/connections/v1/{date}.json"

@dataclass
class ScriptArgs:
    start_date: datetime = datetime.strptime("2023-06-12", "%Y-%m-%d")
    end_date: datetime = datetime.now()
    download: bool = False
    process: bool = False
    download_folder: str = "./connections_data"
    out_file: str = "./connections_prompts.jsonl"
    shuffle: bool = True

async def fetch_and_save(date, download_folder):
    formatted_date = date.strftime("%Y-%m-%d")
    async with httpx.AsyncClient() as client:
        response = await client.get(URL.format(date=formatted_date))
        response_object = response.json()
    pathlib.Path(download_folder).mkdir(parents=True, exist_ok=True)
    with open(f"{download_folder}/{formatted_date}.json", "w", encoding='unicode-escape') as f:
        f.write(json.dumps(response_object, indent=2))

async def download(start_date, end_date, download_folder):
    dates_generated = [
        start_date + timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    tasks = []
    for date in dates_generated:
        tasks.append(fetch_and_save(date, download_folder))
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading data"):
        await f


def process(download_folder, out_file, shuffle=False):
    with open(out_file, "w", encoding='utf-8') as writef:
        for file in sorted(os.listdir(download_folder)): 
            if file.endswith(".json"):
                print(f"Processing {file}")
                with open(f"{download_folder}/{file}", "r", encoding='unicode-escape') as f:
                    data = json.load(f)
                categories = [c for c in data["groups"].keys()]
                
                categories_and_members = {}
                for idx, c in enumerate(categories):
                    category_key = f"category_{idx}"
                    words = [d.lower() for d in data["groups"][c]["members"]]
                    categories_and_members[category_key] = {"words": words, "reason": c.lower()}
                
                all_words = [word for category in categories_and_members.values() for word in category["words"]]
                if shuffle:
                    random.shuffle(all_words)
                out_obj = {
                    "words": all_words,
                    "solution": 
                        {
                            "groups":[categories_and_members['category_1'], categories_and_members['category_0'],
                                      categories_and_members['category_2'], categories_and_members['category_3']
                            ]
                        }
                }
                writef.write(f"{json.dumps(out_obj)}\n")
if __name__ == "__main__":
    args = simple_parsing.parse(ScriptArgs)
    if args.download:
        asyncio.run(download(args.start_date, args.end_date, args.download_folder))
    if args.process:
        process(args.download_folder, args.out_file, args.shuffle)
    print(f"Final output written to {args.out_file}")

