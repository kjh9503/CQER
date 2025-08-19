#!/usr/bin/env python3
"""
Plot entropy by query length from entropy results JSON file
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import re

def count_intersection_queries(query_structure_str):
    """
    Count the number of intersection queries from query structure string
    Example: "(('e', ('r',)), ('e', ('r', 'r', 'r')), ('e', ('r', 'r', 'r')), ('e', ('r', 'r', 'r')), ('e', ('r', 'r', 'r')))" -> 5
    """
    # Count the number of "('e'," patterns which represent individual queries
    count = query_structure_str.count("('e',")
    return count

def plot_entropy_by_length(json_file_path):
    """
    Plot entropy by query length from JSON file
    """
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group entropy values by query length
    length_entropy_map = defaultdict(list)
    
    for query_structure, entropy_values in data['entropy_by_structure'].items():
        # Count intersection queries
        length = count_intersection_queries(query_structure)
        
        # Add all entropy values for this length
        length_entropy_map[length].extend(entropy_values)
    
    # Calculate mean entropy for each length
    lengths = sorted(length_entropy_map.keys())
    mean_entropies = []
    std_entropies = []
    
    for length in lengths:
        entropy_list = length_entropy_map[length]
        mean_entropies.append(np.mean(entropy_list))
        std_entropies.append(np.std(entropy_list))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot with error bars (standard deviation)
    plt.errorbar(lengths, mean_entropies, yerr=std_entropies, 
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    
    # Customize the plot
    plt.xlabel('Number of intersected queries', fontsize=14, fontweight='bold')
    plt.ylabel('Differential Entropy', fontsize=14, fontweight='bold')
    plt.title('Average Differential Entropy by Number of Intersected Queries', 
              fontsize=16, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (length, mean_entropy) in enumerate(zip(lengths, mean_entropies)):
        plt.annotate(f'{mean_entropy:.3f}', 
                    (length, mean_entropy), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=10)
    
    # Set x-axis to show integer values
    plt.xticks(lengths)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'entropy_by_length_plot.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for length, mean_entropy, std_entropy in zip(lengths, mean_entropies, std_entropies):
        count = len(length_entropy_map[length])
        print(f"Length {length}: Mean={mean_entropy:.4f}, Std={std_entropy:.4f}, Count={count}")
    
    return lengths, mean_entropies, std_entropies

def main():
    """
    Main function to find and process entropy results JSON files
    """
    # Look for entropy results JSON files in the current directory
    json_files = [f for f in os.listdir('.') if f.startswith('entropy_results_') and f.endswith('.json')]
    
    if not json_files:
        print("No entropy results JSON files found in current directory.")
        print("Please make sure you're in the logs directory with entropy results files.")
        return
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(json_files)} entropy results file(s):")
    for i, file in enumerate(json_files):
        print(f"{i+1}. {file}")
    
    # Use the most recent file by default
    selected_file = json_files[0]
    print(f"\nUsing most recent file: {selected_file}")
    
    # Plot the data
    try:
        lengths, mean_entropies, std_entropies = plot_entropy_by_length(selected_file)
        print(f"\nSuccessfully processed {selected_file}")
    except Exception as e:
        print(f"Error processing {selected_file}: {e}")

if __name__ == "__main__":
    main()
