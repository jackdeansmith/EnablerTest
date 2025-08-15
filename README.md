# Enabler Test - Does your LLM enable your questionable decisions

Credit to Claude code for almost all the code here.

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