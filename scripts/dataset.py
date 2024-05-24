import os
import json
import random

random.seed(42)

OUT_FILE = "connections_prompts.jsonl"
data_folder = "./connections_data"

def main():
    with open(OUT_FILE, "w", encoding='utf-8') as writef:
        for file in os.listdir(data_folder):
            if file.endswith(".json"):
                with open(f"{data_folder}/{file}", "r", encoding='unicode-escape') as f:
                    data = json.load(f)
                categories = [c for c in data["groups"].keys()]
                categories_and_members = {c.lower(): [d.lower() for d in data["groups"][c]["members"]] for c in categories}
                all_words = [word.lower() for group in categories_and_members.values() for word in group]
                random.shuffle(all_words)
                out_obj = {
                    "words": all_words,
                    "categories": categories_and_members
                }
                writef.write(f"{json.dumps(out_obj)}\n")

if __name__ == "__main__":
    main()