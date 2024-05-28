import os
import json
import random

random.seed(42)

OUT_FILE = "./connections_prompts2.jsonl"
data_folder = "./connections_data"

def main():
    with open(OUT_FILE, "w", encoding='utf-8') as writef:
        for file in os.listdir(data_folder):
            if file.endswith(".json"):
                with open(f"{data_folder}/{file}", "r", encoding='unicode-escape') as f:
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
                    "category_0": categories_and_members['category_0'],
                    "category_1": categories_and_members['category_1'],
                    "category_2": categories_and_members['category_2'],
                    "category_3": categories_and_members['category_3']
                }
                writef.write(f"{json.dumps(out_obj)}\n")

if __name__ == "__main__":
    main()