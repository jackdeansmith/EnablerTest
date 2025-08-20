#!/usr/bin/env python3
"""
Generate a markdown file containing all posts from the opus dataset
with linkable headings for each post version.
"""

import argparse
import csv
import os
from pathlib import Path


def sanitize_heading(text):
    """Convert text to GitHub-compatible heading anchor"""
    # Based on actual GitHub behavior: 
    # "Career Category - Post 1 (Going Back To School) - Red Flag Version"
    # becomes: "career-category---post-1-going-back-to-school---red-flag-version"
    
    # Convert to lowercase
    sanitized = text.lower()
    # Remove parentheses but keep their content
    sanitized = sanitized.replace('(', '').replace(')', '')
    # Replace spaces with hyphens
    sanitized = sanitized.replace(' ', '-')
    # Keep only alphanumeric and hyphens
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '-')
    # GitHub converts sequences like "- -" (space-hyphen-space-hyphen-space) to "---"
    # After space->hyphen conversion, we get "---", but our logic above creates "--"
    # So we need to handle the original " - " pattern specifically
    # Actually, let's restart with the original text and handle " - " specially
    
    # Start over with a different approach
    result = text.lower()
    # Handle the " - " pattern that creates triple hyphens
    result = result.replace(' - ', '---')
    # Remove parentheses
    result = result.replace('(', '').replace(')', '')
    # Replace remaining spaces with hyphens
    result = result.replace(' ', '-')
    # Keep only alphanumeric and hyphens
    result = ''.join(c for c in result if c.isalnum() or c == '-')
    # Clean up any remaining multiple hyphens that aren't triple
    while '----' in result:
        result = result.replace('----', '---')
    while '--' in result and '---' not in result:
        result = result.replace('--', '-')
    return result.strip('-')


def generate_markdown(dataset_dir, output_file):
    """Generate markdown file from dataset directory"""
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Define the order of categories and their display names
    categories = [
        ('career.csv', 'Career'),
        ('finance.csv', 'Finance'), 
        ('relationships.csv', 'Relationships')
    ]
    
    markdown_content = []
    markdown_content.append("# Dataset Posts\n")
    markdown_content.append("This document contains all posts from the opus dataset organized by category.\n")
    markdown_content.append("Each post has both a red flag version (problematic) and reasonable version (improved).\n\n")
    
    # Generate table of contents
    markdown_content.append("## Table of Contents\n")
    for csv_file, category_name in categories:
        csv_path = dataset_path / csv_file
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            markdown_content.append(f"### {category_name} Category\n")
            for i, row in enumerate(rows, 1):
                subcategory = row['subcategory'].title()
                red_flag_anchor = sanitize_heading(f"{category_name} Category - Post {i} ({subcategory}) - Red Flag Version")
                reasonable_anchor = sanitize_heading(f"{category_name} Category - Post {i} ({subcategory}) - Reasonable Version")
                
                markdown_content.append(f"- Post {i} ({subcategory})")
                markdown_content.append(f"  - [Red Flag Version](#{red_flag_anchor})")
                markdown_content.append(f"  - [Reasonable Version](#{reasonable_anchor})")
            markdown_content.append("")
    
    markdown_content.append("---\n")
    
    # Generate the actual posts
    for csv_file, category_name in categories:
        csv_path = dataset_path / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_file} not found in {dataset_dir}")
            continue
            
        markdown_content.append(f"## {category_name} Category\n")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for i, row in enumerate(rows, 1):
            subcategory = row['subcategory'].title()
            
            # Red Flag Version
            markdown_content.append(f"### {category_name} Category - Post {i} ({subcategory}) - Red Flag Version\n")
            markdown_content.append(f"{row['RedFlagPost']}\n")
            markdown_content.append("---\n")
            
            # Reasonable Version  
            markdown_content.append(f"### {category_name} Category - Post {i} ({subcategory}) - Reasonable Version\n")
            markdown_content.append(f"{row['ReasonablePost']}\n")
            markdown_content.append("---\n")
    
    # Write the markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))
    
    print(f"Generated markdown file: {output_file}")
    print(f"Total posts processed: {sum(len(list(csv.DictReader(open(dataset_path / csv_file)))) for csv_file, _ in categories if (dataset_path / csv_file).exists()) * 2} (red flag + reasonable versions)")


def main():
    parser = argparse.ArgumentParser(description='Generate markdown file from opus dataset')
    parser.add_argument('dataset_dir', 
                       help='Path to dataset directory (e.g., data/20250816_anthropic_claude-opus-4.1/)')
    parser.add_argument('-o', '--output', 
                       default='dataset_posts.md',
                       help='Output markdown file name (default: dataset_posts.md)')
    
    args = parser.parse_args()
    
    try:
        generate_markdown(args.dataset_dir, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())