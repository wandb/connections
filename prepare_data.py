import os
import json
import httpx
import pathlib
from dataclasses import dataclass
from tqdm import tqdm
import random
import simple_parsing
from datetime import timedelta, datetime

URL = "https://www.nytimes.com/svc/connections/v1/{date}.json"

@dataclass
class ScriptArgs:
    start_date: datetime = datetime.strptime("2023-06-12", "%Y-%m-%d")
    end_date: datetime = datetime.now()
    seed: int = 42
    download: bool = False
    process: bool = False
    download_folder: str = "./connections_data"
    out_file: str = "./connections_prompts.jsonl"


def download(start_date, end_date, download_folder):
    dates_generated = [
        start_date + timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    for date in tqdm(dates_generated, desc="Downloading data"):
        formatted_date = date.strftime("%Y-%m-%d")
        response = httpx.get(URL.format(date=formatted_date))
        response_object = response.json()
        pathlib.Path(download_folder).mkdir(parents=True, exist_ok=True)
        with open(f"{download_folder}/{formatted_date}.json" , "w", encoding='unicode-escape') as f:
            f.write(json.dumps(response_object, indent=2))


def process(seed, download_folder, out_file):
    random.seed(seed)
    with open(out_file, "w", encoding='utf-8') as writef:
        for file in os.listdir(download_folder):
            if file.endswith(".json"):
                with open(f"{download_folder}/{file}", "r", encoding='unicode-escape') as f:
                    data = json.load(f)
                categories = [c for c in data["groups"].keys()]
                
                categories_and_members = {}
                for idx, c in enumerate(categories):
                    category_key = f"category_{idx}"
                    words = [d.lower() for d in data["groups"][c]["members"]]
                    categories_and_members[category_key] = {"words": words, "reason": c.lower()}
                
                all_words = [word for category in categories_and_members.values() for word in category["words"]]
                random.shuffle(all_words)
                out_obj = {
                    "words": all_words,
                    "solution": 
                        [
                            categories_and_members['category_0'],
                            categories_and_members['category_1'],
                            categories_and_members['category_2'],
                            categories_and_members['category_3']
                        ]
                }
                writef.write(f"{json.dumps(out_obj)}\n")

if __name__ == "__main__":
    args = simple_parsing.parse(ScriptArgs)
    if args.download:
        download(args.start_date, args.end_date, args.download_folder)
    if args.process:
        process(args.seed, args.download_folder, args.out_file)
    print(f"Final output written to {args.out_file}")

