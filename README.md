# Enabler Test: Does Your LLM Enable your Questionable Decisions?

## Project Status and Log

Aug 15: Set up a quick and dirty first draft and ran some quick tests

Aug 16: 
- Significantly improved the synthetic data generation setup and prompts. Generated much more data using the exact same prompts with opus 4.1, gpt-5, and gemini 2.5 pro. Should help with determining if the source of synthetic data contaminates the benchmark.
- Deleted analysis and outputs from previous run, new data is the canonical dataset now and is much bigger, more varied, and higher quality
- Ran evals on all the models I'm interested in on the opus synthetic data
- Ran evals of opus, gpt-5, and gemini 2.5 pro on the gemini and gpt generated datasets as well to determine if model idiosyncratic patterns might pollute the test.
- Should just need to do analysis to get v1 of the test done.


## Project Structure

### Data Directory

The data directory contains datasets organized by name. Each dataset is a directory containing category CSV files. 

```
data/
├── 20250816_anthropic_claude-opus-4.1/   # Dataset name
│   ├── career.csv                        # Category files
│   ├── finance.csv
│   └── relationships.csv
├── 20250816_google_gemini-2.5-pro/       
│   ├── career.csv
│   └── ...
```

### Outputs Directory

The outputs directory is organized with the model being evaluated at the first level and the dataset evaluated against at the second level:

```
outputs/
├── openai_gpt-4o/                                    # Model being evaluated
│   ├── dataset_20250816_anthropic_claude-opus-4.1/   # Dataset evaluated against
│   │   ├── career.csv                                # Raw eval data on this category
│   │   ├── finance.csv
│   │   └── relationships.csv
│   └── dataset_20250816_google_gemini-2.5-pro/       
│       ├── career.csv
│       └── ...
└── anthropic_claude-sonnet-4/                        
    ├── dataset_20250816_anthropic_claude-opus-4.1/   
    └── dataset_20250816_google_gemini-2.5-pro/       
```

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

# Credits

Credit to Claude code for almost all the code here. 

Inspired by Wyatt Walls who [asked models to rank some AI slop math and physics papers on a scale of 1-10](https://x.com/lefthanddraft/status/1955233374605639795) as a way of testing one specific brand of sycophancy and called it "Crank Test". This test is an attempt to do something similar for what I think is probably a more common real world harm. 