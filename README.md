# Enabler Test - Does your LLM enable your questionable decisions

Credit to Claude code for almost all the code here.

## Project Status and Log

Aug 15: Set up a quick and dirty first draft and ran some quick tests

Aug 16: 
- Significantly improved the synthetic data generation setup and prompts. Generated much more data using the exact same prompts with opus 4.1, gpt-5, and gemini 2.5 pro. Should help with determining if the source of synthetic data contaminates the benchmark.
- Deleted analysis and outputs from previous run, new data is the canonical dataset now and is much bigger, more varied, and higher quality


## Setup
```bash
uv sync
export OPENROUTER_API_KEY="your-api-key-here"
```

## Three Steps

### 1. (optional, my synthetic data is checked in) Generate Synthetic Data
```bash
python gen_synthdata.py
```

### 2. Run Model Evaluation
```bash
python eval_model.py
```

### 3. Create Analysis Charts
```bash
python analyze_results.py
```