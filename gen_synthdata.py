#!/usr/bin/env python3
"""
Synthetic data generation script using OpenRouter API
"""

import argparse
import csv
import os
from datetime import datetime
from openai import OpenAI
from prompts import CATEGORIES, REDFLAG_POST_PROMPT, REASONABLE_VERSION_PROMPT
import itertools


def setup_openrouter_client():
    """Initialize OpenRouter client"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def generate_data(client, model, prompt, count):
    """Generate synthetic data using the specified model"""
    results = []
    
    for i in range(count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            generated_text = response.choices[0].message.content
            results.append({
                "id": i + 1,
                "prompt": prompt,
                "generated_text": generated_text
            })
            
            print(f"Generated {i + 1}/{count} samples for category")
            
        except Exception as e:
            print(f"Error generating sample {i + 1}: {e}")
            continue
    
    return results


def save_to_csv(data, output_path, category):
    """Save generated data to CSV file"""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{category}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'subcategory', 'RedFlagPostPrompt', 'RedFlagPost', 'ReasonableVersionPrompt', 'ReasonablePost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} samples to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data using LLMs')
    parser.add_argument('--model', default='anthropic/claude-opus-4.1', 
                       help='Model to use for generation (default: anthropic/claude-opus-4.1)')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of samples to generate per category (default: 10)')
    
    args = parser.parse_args()
    
    # Setup client and output directory
    client = setup_openrouter_client()
    
    # Create output directory with date_modelname format
    date_str = datetime.now().strftime("%Y%m%d")
    model_name = args.model.replace('/', '_')
    output_dir = os.path.join('data', f"{date_str}_{model_name}")
    
    print(f"Generating synthetic data using model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per category: {args.count}")
    
    # Generate data for each category
    for category, category_data in CATEGORIES.items():
        print(f"\nGenerating data for category: {category}")
        
        # Create a cycle iterator for subcategories to rotate through them
        subcategories = category_data.get("subcategories", [""])
        subcategory_cycle = itertools.cycle(subcategories)
        
        all_data = []
        for i in range(args.count):
            subcategory = next(subcategory_cycle)
            # Format the prompt with named parameters
            formatted_prompt = REDFLAG_POST_PROMPT.format(
                category=category_data["category_name"],
                subcategory=subcategory,
            )
            
            try:
                # Set up provider preferences and data collection settings
                if "deepseek" in args.model.lower():
                    extra_body = {
                        "data_collection": "deny",
                        "provider": {
                            "order": ["vertex"],
                            "require_parameters": True
                        }
                    }
                else:
                    extra_body = {
                        "data_collection": "deny"
                    }
                
                # Generate red flag post
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=1.0,
                    extra_body=extra_body
                )
                red_flag_post = response.choices[0].message.content
                
                # Generate reasonable version using the red flag post
                reasonable_prompt = REASONABLE_VERSION_PROMPT.format(original_post=red_flag_post)
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": reasonable_prompt}],
                    temperature=1.0,
                    extra_body=extra_body
                )
                reasonable_post = response.choices[0].message.content
                
                all_data.append({
                    "id": i + 1,
                    "subcategory": subcategory,
                    "RedFlagPostPrompt": formatted_prompt,
                    "RedFlagPost": red_flag_post,
                    "ReasonableVersionPrompt": reasonable_prompt,
                    "ReasonablePost": reasonable_post
                })
                
                print(f"Generated {i + 1}/{args.count} matched pairs for category")
                
            except Exception as e:
                print(f"Error generating sample {i + 1}: {e}")
                continue
        
        save_to_csv(all_data, output_dir, category)
    
    print(f"\nSynthetic data generation complete! Check {output_dir}")


if __name__ == "__main__":
    main()