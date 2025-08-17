# Enabler Test - Does your LLM enable your questionable decisions

Credit to Claude code for almost all the code here.

## Project Status and Log

Aug 15: Set up a quick and dirty first draft and ran some quick tests

Aug 16: 
- Significantly improved the synthetic data generation setup and prompts. Generated much more data using the exact same prompts with opus 4.1, gpt-5, and gemini 2.5 pro. Should help with determining if the source of synthetic data contaminates the benchmark.
- Deleted analysis and outputs from previous run, new data is the canonical dataset now and is much bigger, more varied, and higher quality


## Project Structure

### Data Directory

The data directory contains datasets organized by name. Each dataset is a directory containing category CSV files. You can add your own datasets by creating new directories with category CSV files:

```
data/
├── 20250816_anthropic_claude-opus-4.1/   # Dataset name (can be any name)
│   ├── career.csv                        # Category files
│   ├── finance.csv
│   └── relationships.csv
├── 20250816_google_gemini-2.5-pro/       # Another dataset
│   ├── career.csv
│   └── ...
└── your_custom_dataset/                  # Add your own datasets here
    ├── category1.csv
    └── category2.csv
```

### Outputs Directory

The outputs directory is organized with the model being evaluated at the first level and the dataset evaluated against at the second level:

```
outputs/
├── openai_gpt-4o/                                    # Model being evaluated
│   ├── dataset_20250816_anthropic_claude-opus-4.1/   # Dataset evaluated against
│   │   ├── career.csv
│   │   ├── finance.csv
│   │   └── relationships.csv
│   └── dataset_20250816_google_gemini-2.5-pro/       # Dataset evaluated against
│       ├── career.csv
│       └── ...
└── anthropic_claude-sonnet-4/                        # Model being evaluated
    ├── dataset_20250816_anthropic_claude-opus-4.1/   # Dataset evaluated against
    └── dataset_20250816_google_gemini-2.5-pro/       # Dataset evaluated against
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