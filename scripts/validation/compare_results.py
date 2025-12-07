import sys
import os
import pandas as pd

def load_results(filepath):
    """Loads results from a TSV file (query, target, score)."""
    try:
        # Read first 3 columns: query, target, score
        df = pd.read_csv(filepath, sep='\t', header=None, usecols=[0, 1, 2], names=['query', 'target', 'score'])
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def compare(cpu_file, dpu_file, label):
    print(f"--- Comparison Report: {label} ---")
    cpu_df = load_results(cpu_file)
    dpu_df = load_results(dpu_file)

    print(f"CPU hits: {len(cpu_df)}")
    print(f"DPU hits: {len(dpu_df)}")

    # Create sets of (query, target) pairs
    cpu_pairs = set(zip(cpu_df['query'], cpu_df['target']))
    dpu_pairs = set(zip(dpu_df['query'], dpu_df['target']))

    intersection = cpu_pairs.intersection(dpu_pairs)
    cpu_only = cpu_pairs - dpu_pairs
    dpu_only = dpu_pairs - cpu_pairs

    print(f"Intersection: {len(intersection)}")
    print(f"CPU only: {len(cpu_only)}")
    print(f"DPU only: {len(dpu_only)}")

    if len(cpu_pairs) > 0:
        recall = len(intersection) / len(cpu_pairs)
        print(f"Recall (DPU vs CPU): {recall:.4f}")
    
    if len(dpu_pairs) > 0:
        precision = len(intersection) / len(dpu_pairs)
        print(f"Precision (DPU vs CPU): {precision:.4f}")

    iou = len(intersection) / len(cpu_pairs.union(dpu_pairs)) if len(cpu_pairs.union(dpu_pairs)) > 0 else 0
    print(f"IoU: {iou:.4f}")

    # Compare scores for intersection
    if len(intersection) > 0:
        # Merge on query and target
        merged = pd.merge(cpu_df, dpu_df, on=['query', 'target'], suffixes=('_cpu', '_dpu'))
        
        # Calculate score difference
        merged['diff'] = merged['score_cpu'] - merged['score_dpu']
        
        print(f"Score correlation: {merged['score_cpu'].corr(merged['score_dpu']):.4f}")
        print(f"Max score diff: {merged['diff'].abs().max()}")
        print(f"Mean score diff: {merged['diff'].abs().mean():.4f}")
        
        # Check for significant differences
        significant_diff = merged[merged['diff'].abs() > 1.0] # Tolerance of 1 bit/score
        if not significant_diff.empty:
            print(f"WARNING: {len(significant_diff)} hits have score difference > 1.0")
            print(significant_diff.head())
    
    print("-----------------------------------")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_results.py <cpu_tsv> <dpu_tsv> <label>")
        sys.exit(1)
    
    compare(sys.argv[1], sys.argv[2], sys.argv[3])
