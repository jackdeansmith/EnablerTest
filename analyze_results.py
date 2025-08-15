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


def load_evaluation_data(outputs_dir):
    """Load all evaluation data from outputs directory"""
    all_data = []
    model_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    for model_dir in model_dirs:
        # Extract model name from directory (remove timestamp prefix)
        model_name = "_".join(model_dir.split("_")[2:])  # Skip date_time prefix
        model_path = os.path.join(outputs_dir, model_dir)
        
        # Find all CSV files for this model
        csv_files = glob.glob(os.path.join(model_path, "*.csv"))
        
        for csv_file in csv_files:
            category = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Read CSV and add metadata
            df = pd.read_csv(csv_file)
            df['model'] = model_name
            df['category'] = category
            
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No evaluation data found in {outputs_dir}")
    
    return pd.concat(all_data, ignore_index=True)


def create_box_plots(data, output_dir, category=None):
    """Create box plots comparing red flag vs reasonable scores by model"""
    # Filter data for specific category if provided
    if category:
        plot_data = data[data['category'] == category].copy()
        title_suffix = f" - {category.title()} Category"
        filename_suffix = f"_{category}"
    else:
        plot_data = data.copy()
        title_suffix = " - All Categories"
        filename_suffix = "_all"
    
    # Convert median_score to numeric, handling errors
    plot_data['median_score_numeric'] = pd.to_numeric(plot_data['median_score'], errors='coerce')
    
    # Remove rows where median_score couldn't be converted
    plot_data = plot_data.dropna(subset=['median_score_numeric'])
    
    if plot_data.empty:
        print(f"No valid numeric scores found for {category if category else 'all categories'}")
        return
    
    # Get unique models for plotting
    models = sorted(plot_data['model'].unique())
    
    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        model_data = plot_data[plot_data['model'] == model]
        
        # Create box plot for this model
        red_flag_scores = model_data[model_data['post_type'] == 'red_flag']['median_score_numeric']
        reasonable_scores = model_data[model_data['post_type'] == 'reasonable']['median_score_numeric']
        
        box_data = [red_flag_scores, reasonable_scores]
        box_labels = ['Red Flag Posts', 'Reasonable Posts']
        
        bp = axes[i].boxplot(box_data, labels=box_labels, patch_artist=True, 
                            widths=0.6)
        
        # Color the boxes and lines
        box_colors = ['#ff6b6b', '#4ecdc4']  # Red for red flags, teal for reasonable
        line_colors = ['#ff6b6b', '#4ecdc4']  # Same colors for lines
        
        # Set median line colors and make them wider when quartiles collapse
        for j, median_line in enumerate(bp['medians']):
            median_line.set_color(line_colors[j])
            
            # Check if quartiles are collapsed for this box
            data_values = box_data[j]
            if len(data_values) > 0:
                q1 = data_values.quantile(0.25)
                q3 = data_values.quantile(0.75)
                median_val = data_values.median()
                
                # If quartiles and median are essentially the same, make line thicker
                if abs(q3 - q1) < 0.01 and abs(q1 - median_val) < 0.01:
                    median_line.set_linewidth(6)  # Much thicker line
                else:
                    median_line.set_linewidth(2)  # Normal line
            else:
                median_line.set_linewidth(2)
        
        for j, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[i].set_title(f'{model.replace("_", "/")}')
        axes[i].set_ylabel('Score (1-10)')
        axes[i].set_ylim(0, 11)
        axes[i].grid(True, alpha=0.3)
        
        # Add count annotations
        axes[i].text(1, 10.5, f'n={len(red_flag_scores)}', ha='center', fontsize=9)
        axes[i].text(2, 10.5, f'n={len(reasonable_scores)}', ha='center', fontsize=9)
    
    plt.suptitle(f'Score Distribution by Post Type{title_suffix}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'box_plot{filename_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved box plot: {output_path}")


def generate_summary_stats(data, output_dir):
    """Generate summary statistics"""
    # Convert median_score to numeric
    data['median_score_numeric'] = pd.to_numeric(data['median_score'], errors='coerce')
    clean_data = data.dropna(subset=['median_score_numeric'])
    
    # Calculate summary statistics
    summary = clean_data.groupby(['model', 'category', 'post_type'])['median_score_numeric'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"Saved summary statistics: {summary_path}")
    
    # Also create a pivot table for easier reading
    pivot = clean_data.pivot_table(
        index=['model', 'category'], 
        columns='post_type', 
        values='median_score_numeric',
        aggfunc=['mean', 'count']
    ).round(2)
    
    pivot_path = os.path.join(output_dir, 'pivot_summary.csv')
    pivot.to_csv(pivot_path)
    print(f"Saved pivot summary: {pivot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results and generate plots')
    parser.add_argument('--outputs-dir', required=True, help='Path to outputs directory containing model results')
    parser.add_argument('--analysis-output', default='analysis', help='Directory to save analysis results (default: analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outputs_dir):
        raise ValueError(f"Outputs directory not found: {args.outputs_dir}")
    
    print(f"Loading evaluation data from: {args.outputs_dir}")
    data = load_evaluation_data(args.outputs_dir)
    
    print(f"Loaded {len(data)} evaluation records")
    print(f"Models: {sorted(data['model'].unique())}")
    print(f"Categories: {sorted(data['category'].unique())}")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_stats(data, args.analysis_output)
    
    # Create plots for all categories combined
    print("\nCreating box plots for all categories...")
    create_box_plots(data, args.analysis_output)
    
    # Create plots for each category separately
    categories = sorted(data['category'].unique())
    for category in categories:
        print(f"Creating box plots for {category} category...")
        create_box_plots(data, args.analysis_output, category)
    
    print(f"\nAnalysis complete! Results saved to {args.analysis_output}")


if __name__ == "__main__":
    main()