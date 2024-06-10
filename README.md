# GPT - Connections

Solving [NY Times Connections](https://www.nytimes.com/games/connections) using GPTs...

## Setup
Install dependencies on an environment ideally with pip

```bash
pip install -r requirements.txt
```

## Files

- `connections_prompts.jsonl` - The final consolidated dataset
- `one_shot.py`: A one-shot solution, one prompt, one solution
- `iterative.py`: An iterative solution with feedback (3/4 or 4/4)
- `alpha.py`: A flow engineer solver with planning, inspired by [AlphaCodium paper](https://arxiv.org/pdf/2401.08500)
- `prepare_data.py`: Downloads and preprocesses the dataset (it's already there, only run if you want to download the most recent puzzles)

## RUN
To run on 10 samples:

```python
python one_shot.py --num_samples 10
```

or 

```python
python iterative.py --num_samples 10
```

For the Alpha model, you can select `N` number of initial guess solutions to generate

```python
python alpha.py --N 12 --num_samples 10
```

*Note*: Depending on your OpenAI plan, you may need to tweak the `WEAVE_PARALLELISM` environment variable to avoid rate limits. 
You can do this:

```python
WEAVE_PARALLELISM=5 python alpha.py --num_samples 10
```
