#!/usr/bin/env python3
"""
Synthetic data generation script using OpenRouter API
"""

import argparse
import csv
import os
import random
import asyncio
import aiohttp
from datetime import datetime
from openai import AsyncOpenAI
from prompts import CATEGORIES, GENERATE_SCENARIO, SCENARIO_TO_POST, REASONABLE_VERSION_PROMPT
import itertools
from tqdm.asyncio import tqdm


# Had some trouble with not getting enough variation in the generated scenarios. Injecting some random constraints which seem to help.
FIRST_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SYLLABLE_COUNTS = [1, 2, 3, 4]
US_CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Boston', 'El Paso', 'Nashville', 'Detroit', 'Oklahoma City', 'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore', 'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Kansas City', 'Mesa', 'Atlanta', 'Omaha', 'Colorado Springs', 'Raleigh', 'Virginia Beach', 'Long Beach', 'Miami', 'Oakland', 'Minneapolis', 'Tulsa', 'Bakersfield', 'Wichita', 'Arlington']
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
SIBLING_COUNTS = [0, 1, 2, 3, 4, 5]
PET_OPTIONS = ['Yes', 'No']


MAX_CONCURRENT_REQUESTS = 8

def setup_openrouter_client():
    """Initialize async OpenRouter client"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


async def make_api_request(session, api_key, model, prompt, semaphore, pbar=None):
    """Make a single async API request"""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with semaphore:
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


async def generate_single_sample(session, api_key, model, category_data, subcategory, sample_id, semaphore, pbar=None):
    """Generate a single sample with scenario, red flag post, and reasonable post"""
    try:
        # Stage 1: Generate scenario with random parameters for variation
        scenario_prompt = GENERATE_SCENARIO.format(
            category=category_data["category_name"],
            subcategory=subcategory,
            first_letter_of_first_name=random.choice(FIRST_LETTERS),
            number_of_syllables_in_last_name=random.choice(SYLLABLE_COUNTS),
            us_city=random.choice(US_CITIES),
            season=random.choice(SEASONS),
            number_of_siblings=random.choice(SIBLING_COUNTS),
            do_they_have_a_pet=random.choice(PET_OPTIONS),
        )
        scenario = await make_api_request(session, api_key, model, scenario_prompt, semaphore, pbar)
        
        # Stage 2: Turn scenario into red flag post
        post_prompt = SCENARIO_TO_POST.format(scenario=scenario)
        red_flag_post = await make_api_request(session, api_key, model, post_prompt, semaphore, pbar)
        
        # Generate reasonable version using the red flag post
        reasonable_prompt = REASONABLE_VERSION_PROMPT.format(original_post=red_flag_post)
        reasonable_post = await make_api_request(session, api_key, model, reasonable_prompt, semaphore, pbar)
        
        return {
            "id": sample_id,
            "subcategory": subcategory,
            "Scenario": scenario,
            "RedFlagPost": red_flag_post,
            "ReasonablePost": reasonable_post
        }
        
    except Exception as e:
        print(f"Error generating sample {sample_id}: {e}")
        return None


def save_to_csv(data, output_path, category):
    """Save generated data to CSV file"""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{category}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'subcategory', 'Scenario', 'RedFlagPost', 'ReasonablePost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} samples to {csv_path}")


async def main_async(args):
    # Setup API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    # Create output directory with date_modelname format
    date_str = datetime.now().strftime("%Y%m%d")
    model_name = args.model.replace('/', '_')
    output_dir = os.path.join('data', f"{date_str}_{model_name}")
    
    print(f"Generating synthetic data using model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per category: {args.count}")
    
    # Create session and semaphore for concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        # Generate data for each category
        for category, category_data in CATEGORIES.items():
            print(f"\nGenerating data for category: {category}")
            
            subcategories = category_data.get("subcategories", [""])
            subcategory_cycle = itertools.cycle(subcategories)
            
            tasks = []
            total_api_calls = args.count * 3  # 3 API calls per sample
            
            with tqdm(total=total_api_calls, desc=f"  {category} API calls") as pbar:
                for i in range(args.count):
                    subcategory = next(subcategory_cycle)
                    task = generate_single_sample(
                        session, api_key, args.model, category_data, 
                        subcategory, i + 1, semaphore, pbar
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
            
            # Filter out None results (failed generations)
            all_data = [result for result in results if result is not None]
            
            save_to_csv(all_data, output_dir, category)
    
    print(f"\nSynthetic data generation complete! Check {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data using LLMs')
    parser.add_argument('--model', default='google/gemini-2.5-flash', 
                       help='Model to use for generation (default: google/gemini-2.5-flash)')
    parser.add_argument('--count', type=int, default=30,
                       help='Number of samples to generate per category (default: 30)')
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()