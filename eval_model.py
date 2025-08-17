import argparse
import csv
import os
import glob
from openai import AsyncOpenAI
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from prompts import EVAL_PROMPT, SCORE_EXTRACTION_PROMPT

MODEL_CONFIGS = {
    "claude-opus-4.1": {
        "model": "anthropic/claude-opus-4.1",
        "temperature": 0.3,
        "reasoning": {
            "enabled": False
        }
    },
    "claude-opus-4.1-thinking": {
        "model": "anthropic/claude-opus-4.1",
        "temperature": 0.3,
        "reasoning": {
            "enabled": True
        }
    },
    "claude-sonnet-4": {
        "model": "anthropic/claude-sonnet-4",
        "temperature": 0.3,
        "reasoning": {
            "enabled": False
        }
    },
    "claude-sonnet-4-thinking": {
        "model": "anthropic/claude-sonnet-4",
        "temperature": 0.3,
        "reasoning": {
            "enabled": True 
        }
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "temperature": 0.3,
    },
    "gpt-4o-mini": {
        "model": "openai/gpt-4o-mini", 
        "temperature": 0.3,
    },
    "gpt-5-chat": {
        "model": "openai/gpt-5-chat",
        "temperature": 0.3,
    },
    "gpt-5-thinking-high": {
        "model": "openai/gpt-5",
        "temperature": 0.3,
        "reasoning": {
            "effort": "high"
        }
    },
    "gpt-5-thinking-medium": {
        "model": "openai/gpt-5",
        "temperature": 0.3,
        "reasoning": {
            "effort": "medium"
        }
    },
    "gpt-5-thinking-low": {
        "model": "openai/gpt-5",
        "temperature": 0.3,
        "reasoning": {
            "effort": "low"
        }
    },
    "gpt-5-thinking-minimal": {
        "model": "openai/gpt-5",
        "temperature": 0.3,
        "reasoning": {
            "effort": "minimal"
        }
    },
    "gemini-2.5-pro": {
        "model": "google/gemini-2.5-pro",
        "temperature": 0.3,
        "reasoning": {
            "enabled": True
        }
    },
    "llama-4-maverick": {
        "model": "meta-llama/llama-4-maverick",
        "temperature": 0.3,
    },
    "gemini-2.5-flash-reasoning": {
        "model": "google/gemini-2.5-flash",
        "temperature": 0.3,
        "reasoning": {
            "effort": "high"
        }
    },
    "gemini-2.5-flash": {
        "model": "google/gemini-2.5-flash",
        "temperature": 0.3,
        "reasoning": {
            "enabled": False
        }
    },
}

SCORE_EXTRACTION_CONFIG = {
    "model": "google/gemini-2.0-flash-lite-001",
    "temperature": 0.1,
}

EVAL_CONFIG = {
    "model_aliases": list(MODEL_CONFIGS.keys()),
    "output_base": "outputs", 
    "max_concurrent_requests": 10
}


async def make_api_request(session, api_key, model_config, prompt, semaphore, pbar=None):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **model_config
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with semaphore:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                result = await response.json() if response.status == 200 else None
                if result:
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    return f"HTTP_ERROR_{response.status}: {error_text}"
    except Exception as e:
        return f"REQUEST_ERROR: {str(e)}"
    finally:
        if pbar:
            pbar.update(1)


async def extract_numeric_score_async(session, api_key, response_text, semaphore, pbar=None):
    prompt = SCORE_EXTRACTION_PROMPT.format(response_text=response_text)
    
    score_text = await make_api_request(session, api_key, SCORE_EXTRACTION_CONFIG, prompt, semaphore, pbar)
    
    try:
        # Extract first number found
        import re
        numbers = re.findall(r'\b([1-9]|10)\b', score_text)
        return numbers[0] if numbers else "n/a"
    except Exception as e:
        return "n/a"


async def evaluate_post(session, api_key, model_alias, post_text, semaphore, eval_pbar=None, extract_pbar=None):
    prompt = EVAL_PROMPT.format(post_text=post_text)

    model_config = MODEL_CONFIGS[model_alias]

    # Get single response
    response = await make_api_request(session, api_key, model_config, prompt, semaphore, pbar=eval_pbar)
    
    # Extract score
    score = await extract_numeric_score_async(session, api_key, response, semaphore, pbar=extract_pbar)
    
    return response, score


async def process_single_row(session, api_key, model_alias, row, semaphore, eval_pbar, extract_pbar, row_pbar):
    row_id = row['id']
    subcategory = row['subcategory']
    red_flag_post = row['RedFlagPost']
    reasonable_post = row['ReasonablePost']
    
    # Evaluate both posts concurrently
    red_flag_task = evaluate_post(
        session, api_key, model_alias, red_flag_post, semaphore,
        eval_pbar, extract_pbar
    )
    reasonable_task = evaluate_post(
        session, api_key, model_alias, reasonable_post, semaphore,
        eval_pbar, extract_pbar
    )
    
    (red_flag_response, red_flag_score), (reasonable_response, reasonable_score) = await asyncio.gather(
        red_flag_task, reasonable_task
    )
    
    # Update row progress
    if row_pbar:
        row_pbar.update(1)
    
    # Build single result with both posts
    result = {
        'id': row_id,
        'subcategory': subcategory,
        'red_flag_post': red_flag_post,
        'reasonable_post': reasonable_post,
        'red_flag_response': red_flag_response,
        'reasonable_response': reasonable_response,
        'red_flag_score': red_flag_score,
        'reasonable_score': reasonable_score
    }
    
    return result


async def process_category_csv(session, api_key, model_alias, csv_path, output_dir, category, semaphore, limit_num=None):
    # Read CSV data
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Limit rows if specified
    if limit_num and limit_num > 0:
        rows = rows[:limit_num]
    
    # Calculate total operations for progress bars
    total_rows = len(rows)
    total_evals = total_rows * 2  # 2 posts per row
    total_extractions = total_evals  # One extraction per eval
    
    # Create progress bars for this category
    with tqdm(total=total_rows, desc=f"  post pairs processed in category {category}", position=1, leave=False) as row_pbar:
        with tqdm(total=total_evals, desc=f"  model eval requests in category {category}", position=2, leave=False) as eval_pbar:
            with tqdm(total=total_extractions, desc=f"  score extractions in category {category}", position=3, leave=False) as extract_pbar:
                
                # Process all rows concurrently
                tasks = [
                    process_single_row(session, api_key, model_alias, row, semaphore,
                                     eval_pbar, extract_pbar, row_pbar)
                    for row in rows
                ]
                
                results = await asyncio.gather(*tasks)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{category}.csv")
    
    fieldnames = ['id', 'subcategory', 'red_flag_post', 'reasonable_post', 'red_flag_response', 'reasonable_response', 'red_flag_score', 'reasonable_score']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    return len(results)


async def process_model(session, api_key, model_alias, csv_files, limit_num=None, dataset_name=None):
    output_dir = os.path.join(EVAL_CONFIG['output_base'], model_alias, f"dataset_{dataset_name}")
    
    config = MODEL_CONFIGS[model_alias]
    print(f"\n📊 Processing model alias: {model_alias}")
    print(f"   Model: {config['model']}")
    print(f"   Output: {output_dir}")
    
    # Global semaphore for this model's requests
    semaphore = asyncio.Semaphore(EVAL_CONFIG["max_concurrent_requests"])
    
    # Process all categories with a progress bar
    total_results = 0
    with tqdm(total=len(csv_files), desc=f"Categories for {model_alias}", position=0) as cat_pbar:
        for csv_path in csv_files:
            category = os.path.splitext(os.path.basename(csv_path))[0]
            results_count = await process_category_csv(
                session, api_key, model_alias, csv_path, output_dir, 
                category, semaphore, limit_num
            )
            total_results += results_count
            cat_pbar.update(1)
    
    print(f"✅ Completed {model_alias}: {total_results} results saved to {output_dir}")


async def main_async(data_dir, models_to_run, categories_filter, limit_num):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Extract dataset name from data directory
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    
    # Setup
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    print("\n" + "="*60)
    print("Enabler Test Evaluation")
    print("="*60)
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🤖 Model aliases: {', '.join(models_to_run)}")
    if categories_filter:
        print(f"📂 Categories: {', '.join(categories_filter)}")
    if limit_num:
        print(f"🔢 Limit per category: {limit_num}")
    print(f"⚡ Max concurrent requests: {EVAL_CONFIG['max_concurrent_requests']}")
    print(f"💾 Output base: {EVAL_CONFIG['output_base']}")
    print("="*60)
    
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Filter CSV files by categories if specified
    if categories_filter:
        filtered_files = []
        for category in categories_filter:
            category_file = os.path.join(data_dir, f"{category}.csv")
            if category_file in csv_files:
                filtered_files.append(category_file)
        csv_files = filtered_files
    
    print(f"📋 Processing {len(csv_files)} category files: {', '.join([os.path.basename(f) for f in csv_files])}")
    
    async with aiohttp.ClientSession() as session:

        # Requests within a models eval are concurrent but we evaluate each
        # model in sequence
        for i, model_alias in enumerate(models_to_run, 1):
            print(f"\n[Model {i}/{len(models_to_run)}]")
            await process_model(session, api_key, model_alias, csv_files, limit_num, dataset_name)
    
    print("\n" + "="*60)
    print("🎉 All evaluations complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on Enabler Test')
    parser.add_argument('--data-dir', required=True, help='Path to data directory containing eval data CSV files')
    parser.add_argument('--models', nargs='+', help='Specify which model aliases to run (space-separated). Available aliases: ' + ', '.join(EVAL_CONFIG['model_aliases']))
    parser.add_argument('--categories', nargs='+', help='Specify which categories to run (space-separated). Corresponds to CSV filenames in data set')
    parser.add_argument('--limit-num', type=int, help='Limit number of rows processed per category (useful for testing eval setup quickly)')
    
    args = parser.parse_args()
    
    # Filter model aliases if specified
    if args.models:
        available_aliases = EVAL_CONFIG['model_aliases']
        invalid_aliases = [m for m in args.models if m not in available_aliases]
        if invalid_aliases:
            print(f"❌ Invalid model aliases: {', '.join(invalid_aliases)}")
            print(f"Available aliases: {', '.join(available_aliases)}")
            return
        models_to_run = args.models
    else:
        models_to_run = EVAL_CONFIG['model_aliases']
    
    # Validate categories if specified
    if args.categories:
        csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
        available_categories = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        invalid_categories = [c for c in args.categories if c not in available_categories]
        if invalid_categories:
            print(f"❌ Invalid categories: {', '.join(invalid_categories)}")
            print(f"Available categories: {', '.join(available_categories)}")
            return
    
    # Run the async main function
    asyncio.run(main_async(args.data_dir, models_to_run, args.categories, args.limit_num))


if __name__ == "__main__":
    main()