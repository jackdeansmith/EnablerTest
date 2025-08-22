#!/usr/bin/env python3
"""
Analysis script for evaluation results
"""

import argparse
import csv
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def load_evaluation_data(outputs_dir, dataset_name):
    """Load all evaluation data from outputs directory for a specific dataset"""
    all_data = []
    model_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    for model_dir in model_dirs:
        model_name = model_dir  # Use directory name as model name
        dataset_path = os.path.join(outputs_dir, model_dir, f"dataset_{dataset_name}")
        
        # Skip if this model doesn't have data for this dataset
        if not os.path.exists(dataset_path):
            continue
            
        # Find all CSV files for this model/dataset combination
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        
        for csv_file in csv_files:
            category = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Read CSV and reshape to long format for analysis
            df = pd.read_csv(csv_file)
            
            # Create separate rows for red flag and reasonable posts
            red_flag_rows = []
            reasonable_rows = []
            
            for _, row in df.iterrows():
                # Red flag post row
                red_flag_rows.append({
                    'id': f"{row['id']}_redflag",
                    'original_id': row['id'],
                    'subcategory': row['subcategory'],
                    'post_type': 'red_flag',
                    'post_text': row['red_flag_post'],
                    'response': row['red_flag_response'],
                    'score': row['red_flag_score'],
                    'model': model_name,
                    'category': category
                })
                
                # Reasonable post row
                reasonable_rows.append({
                    'id': f"{row['id']}_reasonable",
                    'original_id': row['id'],
                    'subcategory': row['subcategory'],
                    'post_type': 'reasonable',
                    'post_text': row['reasonable_post'],
                    'response': row['reasonable_response'],
                    'score': row['reasonable_score'],
                    'model': model_name,
                    'category': category
                })
            
            # Combine and add to all_data
            category_df = pd.DataFrame(red_flag_rows + reasonable_rows)
            all_data.append(category_df)
    
    if not all_data:
        raise ValueError(f"No evaluation data found in {outputs_dir}")
    
    return pd.concat(all_data, ignore_index=True)


def create_average_scores_bar_chart(data, output_dir):
    """Create vertical bar chart showing average scores by model and post type"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Calculate average scores by model and post type
    avg_scores = clean_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
    
    # Pivot to have red_flag and reasonable as columns
    pivot_scores = avg_scores.pivot(index='model', columns='post_type', values='score_numeric')
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x = range(len(pivot_scores.index))
    width = 0.35
    
    red_flag_bars = ax.bar([i - width/2 for i in x], pivot_scores['red_flag'], 
                          width, label='Red Flag', color='#ff6b6b', alpha=0.8)
    reasonable_bars = ax.bar([i + width/2 for i in x], pivot_scores['reasonable'], 
                           width, label='Reasonable', color='#4ecdc4', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Score (1-10)')
    ax.set_title('Average Score by Model on All Categories')
    ax.set_xticks(x)
    ax.set_xticklabels([model.replace('_', '/') for model in pivot_scores.index], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar in red_flag_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in reasonable_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved average scores bar chart: {output_path}")


def create_average_scores_by_category(data, output_dir):
    """Create three bar charts showing average scores by model and post type for each category"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Get categories
    categories = sorted(clean_data['category'].unique())
    
    # Create figure with three subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for i, category in enumerate(categories):
        # Filter data for this category
        category_data = clean_data[clean_data['category'] == category]
        
        # Calculate average scores by model and post type
        avg_scores = category_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
        
        # Pivot to have red_flag and reasonable as columns
        pivot_scores = avg_scores.pivot(index='model', columns='post_type', values='score_numeric')
        
        # Create bars for this subplot
        x = range(len(pivot_scores.index))
        width = 0.35
        
        red_flag_bars = axes[i].bar([j - width/2 for j in x], pivot_scores['red_flag'], 
                              width, label='Red Flag', color='#ff6b6b', alpha=0.8)
        reasonable_bars = axes[i].bar([j + width/2 for j in x], pivot_scores['reasonable'], 
                               width, label='Reasonable', color='#4ecdc4', alpha=0.8)
        
        # Customize this subplot
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel('Average Score (1-10)')
        axes[i].set_title(f'{category.title()} Category')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([model.replace('_', '/') for model in pivot_scores.index], 
                               rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_ylim(0, 10)
        
        # Add value labels on bars
        for bar in red_flag_bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in reasonable_bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Average Scores by Model and Category', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved category breakdown chart: {output_path}")


def create_sensitivity_chart(data, output_dir):
    """Create chart showing sensitivity to red flags (difference between reasonable and red flag scores)"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Calculate average scores by model and post type
    avg_scores = clean_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
    
    # Pivot to have red_flag and reasonable as columns
    pivot_scores = avg_scores.pivot(index='model', columns='post_type', values='score_numeric')
    
    # Calculate sensitivity (reasonable - red_flag scores)
    sensitivity = pivot_scores['reasonable'] - pivot_scores['red_flag']
    
    # Create vertical bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by sensitivity for better visualization
    sensitivity_sorted = sensitivity.sort_values(ascending=False)
    
    # Create vertical bars
    bars = ax.bar(range(len(sensitivity_sorted)), sensitivity_sorted.values, 
                  color='#5a9bd4', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel('Model')
    ax.set_ylabel('Sensitivity Score (Reasonable - Red Flag)')
    ax.set_title('Average Sensitivity to Red Flags\nHigher Values = More Sensitive')
    ax.set_xticks(range(len(sensitivity_sorted)))
    ax.set_xticklabels([model.replace('_', '/') for model in sensitivity_sorted.index], 
                       rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, sensitivity_sorted.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.1,
                f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Add a horizontal line at 0 for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_3.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sensitivity chart: {output_path}")


def create_score_distribution_boxplot(data, output_dir):
    """Create vertical box plots showing score distribution for each model, separated by post type"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data = filtered_data.copy()
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get unique models and sort them
    models = sorted(clean_data['model'].unique())
    
    # Create box plot data for red flag and reasonable posts
    red_flag_data = []
    reasonable_data = []
    
    for model in models:
        model_data = clean_data[clean_data['model'] == model]
        red_flag_scores = model_data[model_data['post_type'] == 'red_flag']['score_numeric']
        reasonable_scores = model_data[model_data['post_type'] == 'reasonable']['score_numeric']
        red_flag_data.append(red_flag_scores)
        reasonable_data.append(reasonable_scores)
    
    # Create positions for box plots (two per model)
    positions = []
    for i in range(len(models)):
        positions.extend([i*3 + 1, i*3 + 2])  # Leave space between model groups
    
    # Combine all data for plotting
    all_data = []
    for i in range(len(models)):
        all_data.extend([red_flag_data[i], reasonable_data[i]])
    
    # Create box plot
    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.6)
    
    # Customize box plot colors (red for red flag, teal for reasonable)
    box_colors = ['#ff6b6b', '#4ecdc4'] * len(models)  # Alternate red and teal
    median_colors = ['#cc2d2d', '#2a8b8b'] * len(models)  # Darker red and darker teal
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize median lines to be thicker and appropriately colored
    for median, color in zip(bp['medians'], median_colors):
        median.set_color(color)
        median.set_linewidth(3)
    
    # Customize the chart
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Score Distribution by Model and Post Type')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 10.2)
    
    # Set x-axis ticks and labels
    model_positions = [i*3 + 1.5 for i in range(len(models))]  # Center between the two boxes
    ax.set_xticks(model_positions)
    ax.set_xticklabels([model.replace('_', '/') for model in models], rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff6b6b', alpha=0.7, label='Red Flag'),
                      Patch(facecolor='#4ecdc4', alpha=0.7, label='Reasonable')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_4.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved score distribution box plot: {output_path}")


def create_score_histogram_stack(data, output_dir):
    """Create stacked histograms showing score distribution for each model"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data = filtered_data.copy()
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Get unique models and sort them
    models = sorted(clean_data['model'].unique())
    
    # Create figure with subplots for each model
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3 * len(models)))
    
    # If there's only one model, axes won't be an array
    if len(models) == 1:
        axes = [axes]
    
    # Define bins for histograms (1-10 with 0.5 width bins)
    bins = [i + 0.5 for i in range(11)]  # 0.5, 1.5, 2.5, ..., 10.5
    
    for i, model in enumerate(models):
        model_data = clean_data[clean_data['model'] == model]
        
        # Get scores for each post type
        red_flag_scores = model_data[model_data['post_type'] == 'red_flag']['score_numeric']
        reasonable_scores = model_data[model_data['post_type'] == 'reasonable']['score_numeric']
        
        # Create overlapping histograms
        axes[i].hist(red_flag_scores, bins=bins, alpha=0.7, color='#ff6b6b', 
                    label='Red Flag', density=False, edgecolor='#cc2d2d', linewidth=0.5)
        axes[i].hist(reasonable_scores, bins=bins, alpha=0.7, color='#4ecdc4', 
                    label='Reasonable', density=False, edgecolor='#2a8b8b', linewidth=0.5)
        
        # Customize each subplot
        axes[i].set_title(f'{model.replace("_", "/")}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Count')
        axes[i].set_xlim(0.5, 10.5)
        axes[i].set_ylim(0, 80)
        axes[i].set_xticks(range(1, 11))
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].legend(loc='upper right')
    
    plt.suptitle('Score Distribution Histograms by Model', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_5.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved score histogram stack: {output_path}")


def create_separate_score_charts(data, output_dir):
    """Create two separate bar charts for red flag and reasonable post scores"""
    # Filter out reasoning models (except gpt-5-thinking-high)
    reasoning_models_to_exclude = [
        'claude-opus-4.1-thinking', 'claude-sonnet-4-thinking', 'gemini-2.5-flash-reasoning',
        'gpt-5-thinking-medium', 'gpt-5-thinking-low', 'gpt-5-thinking-minimal'
    ]
    filtered_data = data[~data['model'].isin(reasoning_models_to_exclude)]
    
    # Convert score to numeric and clean data
    filtered_data = filtered_data.copy()
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Calculate average scores by model and post type
    avg_scores = clean_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
    
    # Separate red flag and reasonable scores
    red_flag_scores = avg_scores[avg_scores['post_type'] == 'red_flag'].copy()
    reasonable_scores = avg_scores[avg_scores['post_type'] == 'reasonable'].copy()
    
    # Sort by score in descending order
    red_flag_scores = red_flag_scores.sort_values('score_numeric', ascending=False)
    reasonable_scores = reasonable_scores.sort_values('score_numeric', ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Red flag scores chart
    bars1 = ax1.bar(range(len(red_flag_scores)), red_flag_scores['score_numeric'], 
                    color='#ff6b6b', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Score (1-10)')
    ax1.set_title('Average Red Flag Post Scores (sorted)')
    ax1.set_xticks(range(len(red_flag_scores)))
    ax1.set_xticklabels([model.replace('_', '/') for model in red_flag_scores['model']], 
                        rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, value in zip(bars1, red_flag_scores['score_numeric']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Reasonable scores chart
    bars2 = ax2.bar(range(len(reasonable_scores)), reasonable_scores['score_numeric'], 
                    color='#4ecdc4', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Score (1-10)')
    ax2.set_title('Average Reasonable Post Scores (sorted)')
    ax2.set_xticks(range(len(reasonable_scores)))
    ax2.set_xticklabels([model.replace('_', '/') for model in reasonable_scores['model']], 
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, value in zip(bars2, reasonable_scores['score_numeric']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_6.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved separate score charts: {output_path}")


def create_reasoning_model_series_chart(data, output_dir):
    """Create chart showing reasoning model series from low to high reasoning"""
    # Convert score to numeric and clean data
    data['score_numeric'] = pd.to_numeric(data['score'], errors='coerce')
    clean_data = data.dropna(subset=['score_numeric'])
    
    # Define reasoning model series (low to high reasoning)
    reasoning_series = {
        'Claude Opus 4.1': ['claude-opus-4.1', 'claude-opus-4.1-thinking'],
        'Claude Sonnet 4': ['claude-sonnet-4', 'claude-sonnet-4-thinking'],
        'Gemini 2.5 Flash': ['gemini-2.5-flash', 'gemini-2.5-flash-reasoning'],
        'GPT-5 Thinking': ['gpt-5-thinking-minimal', 'gpt-5-thinking-low', 'gpt-5-thinking-medium', 'gpt-5-thinking-high']
    }
    
    # Filter to only include models in our reasoning series
    all_reasoning_models = []
    for series in reasoning_series.values():
        all_reasoning_models.extend(series)
    
    reasoning_data = clean_data[clean_data['model'].isin(all_reasoning_models)]
    
    # Calculate average scores by model and post type
    avg_scores = reasoning_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
    
    # Create figure with subplots for each series
    fig, axes = plt.subplots(len(reasoning_series), 1, figsize=(12, 3 * len(reasoning_series)))
    
    # If there's only one series, axes won't be an array
    if len(reasoning_series) == 1:
        axes = [axes]
    
    for i, (series_name, models_in_series) in enumerate(reasoning_series.items()):
        # Filter data for this series
        series_data = avg_scores[avg_scores['model'].isin(models_in_series)]
        
        # Skip if no data for this series
        if series_data.empty:
            axes[i].text(0.5, 0.5, f'No data for {series_name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(series_name)
            continue
        
        # Pivot to have red_flag and reasonable as columns
        pivot_scores = series_data.pivot(index='model', columns='post_type', values='score_numeric')
        
        # Reorder models according to the reasoning series order
        ordered_models = [model for model in models_in_series if model in pivot_scores.index]
        pivot_scores = pivot_scores.reindex(ordered_models)
        
        # Create bars
        x = range(len(pivot_scores.index))
        width = 0.35
        
        red_flag_bars = axes[i].bar([j - width/2 for j in x], pivot_scores['red_flag'], 
                                   width, label='Red Flag', color='#ff6b6b', alpha=0.8)
        reasonable_bars = axes[i].bar([j + width/2 for j in x], pivot_scores['reasonable'], 
                                     width, label='Reasonable', color='#4ecdc4', alpha=0.8)
        
        # Customize this subplot
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Average Score (1-10)')
        axes[i].set_title(series_name)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([model.replace('_', '/') for model in pivot_scores.index])
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_ylim(0, 10)
        
        # Add value labels on bars
        for bar in red_flag_bars:
            height = bar.get_height()
            if not pd.isna(height):
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in reasonable_bars:
            height = bar.get_height()
            if not pd.isna(height):
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Performance Within Reasoning Model Series', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_7.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reasoning model series chart: {output_path}")


def create_cross_dataset_bias_chart(outputs_dir):
    """Create chart showing potential bias from synthetic data source models"""
    # Define the datasets and their corresponding models to evaluate
    datasets_info = {
        '20250816_anthropic_claude-opus-4.1': 'Claude Opus 4.1 Generated Data',
        '20250816_google_gemini-2.5-pro': 'Gemini 2.5 Pro Generated Data', 
        '20250816_openai_gpt-5': 'GPT-5 Generated Data'
    }
    
    # Models to evaluate (the three data generators)
    models_to_evaluate = ['claude-opus-4.1', 'gemini-2.5-pro', 'gpt-5-thinking-high']
    
    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(len(datasets_info), 1, figsize=(12, 4 * len(datasets_info)))
    
    # If there's only one dataset, axes won't be an array
    if len(datasets_info) == 1:
        axes = [axes]
    
    for i, (dataset_name, display_name) in enumerate(datasets_info.items()):
        try:
            # Load data for this specific dataset
            dataset_data = load_evaluation_data(outputs_dir, dataset_name)
            
            # Filter to only the three models we want to compare
            filtered_data = dataset_data[dataset_data['model'].isin(models_to_evaluate)]
            
            if filtered_data.empty:
                axes[i].text(0.5, 0.5, f'No data for {display_name}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(display_name)
                continue
            
            # Convert score to numeric and clean data
            filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
            clean_data = filtered_data.dropna(subset=['score_numeric'])
            
            # Calculate average scores by model and post type
            avg_scores = clean_data.groupby(['model', 'post_type'])['score_numeric'].mean().reset_index()
            
            # Pivot to have red_flag and reasonable as columns
            pivot_scores = avg_scores.pivot(index='model', columns='post_type', values='score_numeric')
            
            # Reorder models to consistent order
            ordered_models = [model for model in models_to_evaluate if model in pivot_scores.index]
            pivot_scores = pivot_scores.reindex(ordered_models)
            
            # Create bars
            x = range(len(pivot_scores.index))
            width = 0.35
            
            red_flag_bars = axes[i].bar([j - width/2 for j in x], pivot_scores['red_flag'], 
                                       width, label='Red Flag', color='#ff6b6b', alpha=0.8)
            reasonable_bars = axes[i].bar([j + width/2 for j in x], pivot_scores['reasonable'], 
                                         width, label='Reasonable', color='#4ecdc4', alpha=0.8)
            
            # Customize this subplot
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Average Score (1-10)')
            axes[i].set_title(display_name)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([model.replace('_', '/').replace('-thinking-high', '') for model in pivot_scores.index])
            axes[i].legend()
            axes[i].grid(True, alpha=0.3, axis='y')
            axes[i].set_ylim(0, 10)
            
            # Add value labels on bars
            for bar in red_flag_bars:
                height = bar.get_height()
                if not pd.isna(height):
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            
            for bar in reasonable_bars:
                height = bar.get_height()
                if not pd.isna(height):
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error loading {display_name}: {str(e)}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(display_name)
    
    plt.suptitle('Potential Bias from Synthetic Data Source', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the chart
    output_path = os.path.join('analysis', 'figure_8.png')
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-dataset bias chart: {output_path}")


def create_selected_models_distribution_histogram(data, output_dir):
    """Create distribution histogram comparing only Claude 4 Sonnet, GPT-4o, GPT-5-chat, and LLama 4 Maverick"""
    # Define the specific models to include and their official display names
    selected_models = ['claude-sonnet-4', 'gpt-4o', 'gpt-5-chat', 'llama-4-maverick']
    
    # Map to official model names from eval_model.py MODEL_CONFIGS
    model_display_names = {
        'claude-sonnet-4': 'Claude Sonnet 4',
        'gpt-4o': 'GPT-4o',
        'gpt-5-chat': 'GPT-5 Chat',
        'llama-4-maverick': 'Llama 4 Maverick'
    }
    
    # Filter data to only include the selected models
    filtered_data = data[data['model'].isin(selected_models)]
    
    if filtered_data.empty:
        print("Warning: No data found for the selected models (Claude 4 Sonnet, GPT-4o, GPT-5-chat, LLama 4 Maverick)")
        return
    
    # Convert score to numeric and clean data
    filtered_data = filtered_data.copy()
    filtered_data['score_numeric'] = pd.to_numeric(filtered_data['score'], errors='coerce')
    clean_data = filtered_data.dropna(subset=['score_numeric'])
    
    # Get models that actually exist in the data
    available_models = sorted([model for model in selected_models if model in clean_data['model'].unique()])
    
    if not available_models:
        print("Warning: None of the selected models found in data")
        return
    
    # Create figure with subplots for each model
    fig, axes = plt.subplots(len(available_models), 1, figsize=(12, 3 * len(available_models)))
    
    # If there's only one model, axes won't be an array
    if len(available_models) == 1:
        axes = [axes]
    
    # Define bins for histograms (1-10 with 0.5 width bins)
    bins = [i + 0.5 for i in range(11)]  # 0.5, 1.5, 2.5, ..., 10.5
    
    for i, model in enumerate(available_models):
        model_data = clean_data[clean_data['model'] == model]
        
        # Get scores for each post type
        red_flag_scores = model_data[model_data['post_type'] == 'red_flag']['score_numeric']
        reasonable_scores = model_data[model_data['post_type'] == 'reasonable']['score_numeric']
        
        # Create overlapping histograms
        axes[i].hist(red_flag_scores, bins=bins, alpha=0.7, color='#ff6b6b', 
                    label='Red Flag', density=False, edgecolor='#cc2d2d', linewidth=0.5)
        axes[i].hist(reasonable_scores, bins=bins, alpha=0.7, color='#4ecdc4', 
                    label='Reasonable', density=False, edgecolor='#2a8b8b', linewidth=0.5)
        
        # Customize each subplot using official model names
        axes[i].set_title(model_display_names.get(model, model))
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Count')
        axes[i].set_xlim(0.5, 10.5)
        axes[i].set_ylim(0, 80)
        axes[i].set_xticks(range(1, 11))
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].legend(loc='upper right')
    
    # Remove the suptitle entirely
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'figure_9.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved selected models distribution histogram: {output_path}")
    print(f"Models included: {', '.join(available_models)}")


# Analysis functions will be added here as requested


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results and generate plots')
    parser.add_argument('--outputs-dir', required=True, help='Path to outputs directory containing model results')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to analyze (e.g., "20250816_anthropic_claude-opus-4.1")')
    parser.add_argument('--analysis-output', default='analysis', help='Directory to save analysis results (default: analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outputs_dir):
        raise ValueError(f"Outputs directory not found: {args.outputs_dir}")
    
    print(f"Loading evaluation data from: {args.outputs_dir}")
    print(f"Dataset: {args.dataset_name}")
    data = load_evaluation_data(args.outputs_dir, args.dataset_name)
    
    print(f"Loaded {len(data)} evaluation records")
    print(f"Models: {sorted(data['model'].unique())}")
    print(f"Categories: {sorted(data['category'].unique())}")
    
    # Create output directory
    os.makedirs(args.analysis_output, exist_ok=True)
    
    # Create average scores bar chart
    print("\nCreating average scores bar chart...")
    create_average_scores_bar_chart(data, args.analysis_output)
    
    # Create average scores by category chart
    print("\nCreating average scores by category chart...")
    create_average_scores_by_category(data, args.analysis_output)
    
    # Create sensitivity chart
    print("\nCreating sensitivity chart...")
    create_sensitivity_chart(data, args.analysis_output)
    
    # Create score distribution box plot
    print("\nCreating score distribution box plot...")
    create_score_distribution_boxplot(data, args.analysis_output)
    
    # Create score histogram stack
    print("\nCreating score histogram stack...")
    create_score_histogram_stack(data, args.analysis_output)
    
    # Create separate score charts
    print("\nCreating separate score charts...")
    create_separate_score_charts(data, args.analysis_output)
    
    # Create reasoning model series chart
    print("\nCreating reasoning model series chart...")
    create_reasoning_model_series_chart(data, args.analysis_output)
    
    # Create cross-dataset bias chart
    print("\nCreating cross-dataset bias chart...")
    create_cross_dataset_bias_chart(args.outputs_dir)
    
    # Create selected models distribution histogram
    print("\nCreating selected models distribution histogram...")
    create_selected_models_distribution_histogram(data, args.analysis_output)
    
    print(f"\nAnalysis complete! Results saved to {args.analysis_output}")


if __name__ == "__main__":
    main()