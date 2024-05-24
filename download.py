import json
import httpx
import pathlib
from tqdm import tqdm
from datetime import timedelta, datetime

url = "https://www.nytimes.com/svc/connections/v1/{date}.json"
start_date = datetime.strptime("2023-06-12", "%Y-%m-%d")
data_folder = "./connections_data"

def main():
    end_date = datetime.now()
    dates_generated = [
        start_date + timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    for date in tqdm(dates_generated):
        formatted_date = date.strftime("%Y-%m-%d")
        response = httpx.get(url.format(date=formatted_date))
        response_object = response.json()
        pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
        with open(f"{data_folder}/{formatted_date}.json" , "w", encoding='unicode-escape') as f:
            f.write(json.dumps(response_object, indent=2))


if __name__ == "__main__":
    main()