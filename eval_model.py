"""
Evaluation script for testing models on synthetic reddit posts
"""

import argparse
import csv
import os
import glob
import time
from datetime import datetime
from openai import AsyncOpenAI
import asyncio
import aiohttp
import json
from tqdm.asyncio import tqdm
from prompts import EVAL_PROMPT, SCORE_EXTRACTION_PROMPT


# Configuration - Edit these to change evaluation settings
EVAL_CONFIG = {
    "models": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-5-chat",
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-opus-4.1",
        "google/gemini-2.5-pro",
    ],
    "num_evals": 3,
    "output_base": "outputs",
    "max_concurrent_requests": 10
}


def setup_openrouter_client():
    """Initialize async OpenRouter client"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


async def make_api_request(session, api_key, model, prompt, request_type="eval", pbar=None):
    """Make a single API request"""
    extra_body = {"data_collection": "deny"}
    if "deepseek" in model.lower():
        extra_body["provider"] = {
            "order": ["vertex"],
            "require_parameters": True
        }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3 if request_type == "eval" else 0.1,
        **extra_body
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status == 200:
                result = await response.json()
                if pbar:
                    pbar.update(1)
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                if pbar:
                    pbar.update(1)
                return f"HTTP_ERROR_{response.status}: {error_text}"
    except Exception as e:
        if pbar:
            pbar.update(1)
        return f"REQUEST_ERROR: {str(e)}"


async def extract_numeric_score_async(session, api_key, response_text, semaphore, pbar=None):
    """Extract numeric score from response using Gemini Flash"""
    prompt = SCORE_EXTRACTION_PROMPT.format(response_text=response_text)
    
    async with semaphore:
        score_text = await make_api_request(
            session, 
            api_key,
            "google/gemini-2.0-flash-lite-001",
            prompt,
            request_type="extraction",
            pbar=pbar
        )
    
    try:
        # Extract first number found
        import re
        numbers = re.findall(r'\b([1-9]|10)\b', score_text)
        return numbers[0] if numbers else "PARSE_ERROR"
    except Exception as e:
        return "EXTRACTION_ERROR"


async def evaluate_post(session, api_key, model, post_text, num_evals, semaphore, post_label, eval_pbar=None, extract_pbar=None):
    """Evaluate a single post multiple times"""
    prompt = EVAL_PROMPT.format(post_text=post_text)
    
    # Get all responses concurrently
    tasks = []
    for i in range(num_evals):
        async def bounded_request():
            async with semaphore:
                return await make_api_request(session, api_key, model, prompt, request_type="eval", pbar=eval_pbar)
        tasks.append(bounded_request())
    
    responses = await asyncio.gather(*tasks)
    
    # Extract scores concurrently
    score_tasks = [
        extract_numeric_score_async(session, api_key, response, semaphore, pbar=extract_pbar)
        for response in responses
    ]
    scores = await asyncio.gather(*score_tasks)
    
    return responses, scores


async def process_single_row(session, api_key, model, row, num_evals, semaphore, category, eval_pbar, extract_pbar, row_pbar):
    """Process a single row from CSV"""
    row_id = row['id']
    subcategory = row['subcategory']
    red_flag_post = row['RedFlagPost']
    reasonable_post = row['ReasonablePost']
    
    # Evaluate both posts concurrently
    red_flag_task = evaluate_post(
        session, api_key, model, red_flag_post, num_evals, semaphore,
        f"{category} #{row_id} red flag post",
        eval_pbar, extract_pbar
    )
    reasonable_task = evaluate_post(
        session, api_key, model, reasonable_post, num_evals, semaphore,
        f"{category} #{row_id} reasonable post",
        eval_pbar, extract_pbar
    )
    
    (red_flag_responses, red_flag_scores), (reasonable_responses, reasonable_scores) = await asyncio.gather(
        red_flag_task, reasonable_task
    )
    
    # Update row progress
    if row_pbar:
        row_pbar.update(1)
    
    # Calculate median scores
    def calculate_median(scores):
        numeric_scores = []
        for score in scores:
            try:
                numeric_scores.append(float(score))
            except (ValueError, TypeError):
                continue
        if numeric_scores:
            numeric_scores.sort()
            n = len(numeric_scores)
            if n % 2 == 0:
                return (numeric_scores[n//2 - 1] + numeric_scores[n//2]) / 2
            else:
                return numeric_scores[n//2]
        return "NO_VALID_SCORES"
    
    red_flag_median = calculate_median(red_flag_scores)
    reasonable_median = calculate_median(reasonable_scores)
    
    # Build results
    red_flag_prompt = EVAL_PROMPT.format(post_text=red_flag_post)
    reasonable_prompt = EVAL_PROMPT.format(post_text=reasonable_post)
    
    red_flag_result = {
        'id': f"{row_id}_redflag",
        'original_id': row_id,
        'subcategory': subcategory,
        'post_type': 'red_flag',
        'post_text': red_flag_post,
        'eval_prompt': red_flag_prompt,
        'median_score': red_flag_median
    }
    
    reasonable_result = {
        'id': f"{row_id}_reasonable",
        'original_id': row_id,
        'subcategory': subcategory,
        'post_type': 'reasonable',
        'post_text': reasonable_post,
        'eval_prompt': reasonable_prompt,
        'median_score': reasonable_median
    }
    
    # Add individual evaluation columns
    for i in range(num_evals):
        red_flag_result[f'response_{i+1}'] = red_flag_responses[i]
        red_flag_result[f'score_{i+1}'] = red_flag_scores[i]
        reasonable_result[f'response_{i+1}'] = reasonable_responses[i]
        reasonable_result[f'score_{i+1}'] = reasonable_scores[i]
    
    return [red_flag_result, reasonable_result]


async def process_category_csv(session, api_key, model, csv_path, output_dir, category, num_evals, semaphore):
    """Process a single category CSV file"""
    # Read CSV data
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Calculate total operations for progress bars
    total_rows = len(rows)
    total_evals = total_rows * 2 * num_evals  # 2 posts per row, num_evals per post
    total_extractions = total_evals  # One extraction per eval
    
    # Create progress bars for this category
    with tqdm(total=total_rows, desc=f"  {category} rows", position=1, leave=False) as row_pbar:
        with tqdm(total=total_evals, desc=f"  {category} evals", position=2, leave=False) as eval_pbar:
            with tqdm(total=total_extractions, desc=f"  {category} extractions", position=3, leave=False) as extract_pbar:
                
                # Process all rows concurrently
                tasks = [
                    process_single_row(session, api_key, model, row, num_evals, semaphore, category, 
                                     eval_pbar, extract_pbar, row_pbar)
                    for row in rows
                ]
                
                results_nested = await asyncio.gather(*tasks)
    
    # Flatten results
    results = []
    for row_results in results_nested:
        results.extend(row_results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{category}.csv")
    
    # Create dynamic fieldnames
    base_fieldnames = ['id', 'original_id', 'subcategory', 'post_type', 'post_text', 'eval_prompt', 'median_score']
    eval_fieldnames = []
    for i in range(num_evals):
        eval_fieldnames.extend([f'response_{i+1}', f'score_{i+1}'])
    fieldnames = base_fieldnames + eval_fieldnames
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    return len(results)


async def process_model(session, api_key, model, csv_files):
    """Process all categories for a single model"""
    # Create output directory for this model
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace('/', '_')
    output_dir = os.path.join(EVAL_CONFIG['output_base'], f"{date_str}_{model_name}")
    
    print(f"\n📊 Processing model: {model}")
    print(f"   Output: {output_dir}")
    
    # Global semaphore for this model's requests
    semaphore = asyncio.Semaphore(EVAL_CONFIG["max_concurrent_requests"])
    
    # Process all categories with a progress bar
    total_results = 0
    with tqdm(total=len(csv_files), desc=f"Categories for {model}", position=0) as cat_pbar:
        for csv_path in csv_files:
            category = os.path.splitext(os.path.basename(csv_path))[0]
            results_count = await process_category_csv(
                session, api_key, model, csv_path, output_dir, 
                category, EVAL_CONFIG['num_evals'], semaphore
            )
            total_results += results_count
            cat_pbar.update(1)
    
    print(f"✅ Completed {model}: {total_results} results saved to {output_dir}")


async def main_async(data_dir):
    """Main async function"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Setup
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    print("\n" + "="*60)
    print("🚀 Reddit Post Evaluation Script")
    print("="*60)
    print(f"📁 Data directory: {data_dir}")
    print(f"🤖 Models: {', '.join(EVAL_CONFIG['models'])}")
    print(f"🔢 Evaluations per post: {EVAL_CONFIG['num_evals']}")
    print(f"⚡ Max concurrent requests: {EVAL_CONFIG['max_concurrent_requests']}")
    print(f"💾 Output directory: {EVAL_CONFIG['output_base']}")
    print("="*60)
    
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f"📋 Found {len(csv_files)} category files: {', '.join([os.path.basename(f) for f in csv_files])}")
    
    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        # Process each model sequentially (but all posts within a model are concurrent)
        for i, model in enumerate(EVAL_CONFIG['models'], 1):
            print(f"\n[Model {i}/{len(EVAL_CONFIG['models'])}]")
            await process_model(session, api_key, model, csv_files)
    
    print("\n" + "="*60)
    print("🎉 All evaluations complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on synthetic reddit posts')
    parser.add_argument('--data-dir', required=True, help='Path to data directory containing category CSVs')
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main_async(args.data_dir))


if __name__ == "__main__":
    main()