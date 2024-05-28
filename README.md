# GPT - Connections

Solving [NY Times Connections](https://www.nytimes.com/games/connections) using GPTs...


## Files

- prepare_data.py: Downloads and process the dataset

```python
python prepare_data.py --download --process
```

- connections_prompts.jsonl - The final consolidated dataset
- `run.py`: An iterative solution with feedback (3/4 or 4/4)
- `alpha.py`: A flow engineer with plan
