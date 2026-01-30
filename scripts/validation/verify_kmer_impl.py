#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
from collections import defaultdict

# ==============================================================================
# 1. SETUP & PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Detailed DPU vs CPU K-mer Verification")
    parser.add_argument("cpu_tsv", help="Path to CPU results TSV")
    parser.add_argument("dpu_tsv", help="Path to DPU results TSV")
    parser.add_argument("--query", help="Query FASTA (required for verification)")
    parser.add_argument("--target", help="Target FASTA (required for verification)")
    parser.add_argument("--kmer", type=int, default=7, help="K-mer size")
    parser.add_argument("--mask", type=str, default="1111111", help="Binary mask pattern")
    parser.add_argument("--no-diag-check", action="store_true", help="Disable expensive verification")
    return parser.parse_args()

def parse_fasta(filepath):
    seqs = {}
    if not filepath: return seqs
    name = None
    seq_parts = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if name: seqs[name] = ''.join(seq_parts)
                    raw_header = line[1:].split()[0]
                    if '|' in raw_header:
                        name = raw_header.split('|')[1] 
                    else:
                        name = raw_header
                    seq_parts = []
                else:
                    seq_parts.append(line)
        if name: seqs[name] = ''.join(seq_parts)
    except Exception as e:
        print(f"Error reading FASTA {filepath}: {e}")
        sys.exit(1)
    return seqs

def load_tsv_for_metrics(filepath):
    """
    Loads TSV into Sets for Precision/Recall/IoU calculation.
    Returns:
      set_qt: set of (q, t)
      set_qtd: set of (q, t, d)
      lookup_qt_to_diags: dict[(q,t)] -> list of diagonals
    """
    set_qt = set()
    set_qtd = set()
    lookup = defaultdict(list)
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0].lower() == "query": continue 
                
                q, t = parts[0], parts[1]
                diag = int(parts[3]) if len(parts) > 3 else 0
                
                set_qt.add((q, t))
                set_qtd.add((q, t, diag))
                lookup[(q, t)].append(diag)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
        
    return set_qt, set_qtd, lookup

# ==============================================================================
# 2. PANDAS SCORE ANALYSIS
# ==============================================================================

def analyze_scores_pandas(cpu_path, dpu_path):
    print("\n-----------------------------------")
    print("      Score Analysis (Pandas)      ")
    print("-----------------------------------")
    
    def load_and_prep(path, label):
        # Load TSV. Assuming no header, or handling header if present.
        # Try reading first line to check header
        try:
            df = pd.read_csv(path, sep='\t', header=None)
            # Check if first row is header
            if isinstance(df.iloc[0,0], str) and df.iloc[0,0].lower() == 'query':
                df = pd.read_csv(path, sep='\t', header=0)
            else:
                # Assign default columns
                # Cols: query, target, score, diagonal (optional)
                cols = ['query', 'target', 'score']
                if df.shape[1] >= 4: cols.append('diagonal')
                if df.shape[1] > 4: cols.extend([f'col_{i}' for i in range(4, df.shape[1])])
                df.columns = cols
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()

        # 1. Cap scores at 255 BEFORE grouping (Input Sanitization)
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['score'] = df['score'].clip(upper=255)

        # 2. Group by Query-Target and Sum (Aggregation)
        # This merges multiple diagonals for the same pair into one score
        grouped = df.groupby(['query', 'target'], as_index=False)['score'].sum()

        # 3. Cap aggregated scores at 255 again 
        # (Essential to compare against MMseqs2 CPU output which is saturated)
        grouped['score'] = grouped['score'].clip(upper=255)
        
        return grouped

    a_df = load_and_prep(cpu_path, "CPU")
    b_df = load_and_prep(dpu_path, "DPU")

    if a_df.empty or b_df.empty:
        print("Dataframes empty, skipping score analysis.")
        return

    # Find Intersection
    # We perform an inner merge to find the intersection of keys
    merged = pd.merge(a_df, b_df, on=['query', 'target'], suffixes=('_a', '_b'))
    
    print(f"Intersection for Score Comparison: {len(merged)} pairs")

    if not merged.empty:
        # Calculate Diff
        merged['diff'] = merged['score_a'] - merged['score_b']

        # Correlation
        corr = merged['score_a'].corr(merged['score_b'])
        print(f"Score correlation: {corr:.4f}" if pd.notna(corr) else "Score correlation: NA")
        print(f"Max score diff: {merged['diff'].abs().max()}" )
        print(f"Mean score diff: {merged['diff'].abs().mean():.4f}")

        # Significant Differences
        significant_diff = merged[merged['diff'].abs() > 1.0]  # Tolerance of 1 bit/score
        if not significant_diff.empty:
            significant_diff = significant_diff.reindex(significant_diff['diff'].abs().sort_values(ascending=False).index)
            print(f"WARNING: {len(significant_diff)} hits have score difference > 1.0")
            cols = ['query', 'target', 'score_a', 'score_b', 'diff']
            print(significant_diff[cols].head())
        else:
            print("Scores match perfectly (within tolerance).")

    print("-----------------------------------")

# ==============================================================================
# 3. GEOMETRIC VERIFICATION LOGIC
# ==============================================================================

def check_diagonal_validity(q_seq, t_seq, diag, kmer, offsets):
    q_len = len(q_seq)
    t_len = len(t_seq)
    
    start_i = max(0, diag)
    end_i = min(q_len, t_len + diag)
    
    hits_found = 0
    k = kmer
    search_end = end_i - k + 1
    
    if search_end <= start_i:
        return False
        
    for i in range(start_i, search_end):
        j = i - diag
        
        match = True
        for off in offsets:
            if q_seq[i+off] != t_seq[j+off]:
                match = False
                break
        
        if match:
            hits_found += 1
            if hits_found >= 2:
                return True
                
    return False

def analyze_subset(subset, label, granularity, lookup, queries, targets, kmer, mask_offsets):
    if not subset:
        return 0, 0

    valid_count = 0
    invalid_count = 0
    
    for item in subset:
        is_valid = False
        
        if granularity == "Diagonal":
            q, t, d = item
            if q in queries and t in targets:
                if check_diagonal_validity(queries[q], targets[t], d, kmer, mask_offsets):
                    is_valid = True
        
        elif granularity == "Sequence":
            q, t = item
            if q in queries and t in targets:
                diags = lookup.get((q,t), [])
                for d in diags:
                    if check_diagonal_validity(queries[q], targets[t], d, kmer, mask_offsets):
                        is_valid = True
                        break 
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            
    return valid_count, invalid_count

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    
    print("Loading TSV files for Metrics...")
    cpu_qt, cpu_qtd, cpu_lookup = load_tsv_for_metrics(args.cpu_tsv)
    dpu_qt, dpu_qtd, dpu_lookup = load_tsv_for_metrics(args.dpu_tsv)
    
    mask_offsets = [i for i, c in enumerate(args.mask) if c == '1']
    
    queries = {}
    targets = {}
    if not args.no_diag_check:
        print("Loading FASTA files...")
        queries = parse_fasta(args.query)
        targets = parse_fasta(args.target)

    # Helper for Metrics Block
    def run_analysis_block(set_a, set_b, lookup_a, lookup_b, label, granularity):
        intersection = len(set_a & set_b)
        len_a = len(set_a)
        len_b = len(set_b)
        
        a_only = set_a - set_b
        b_only = set_b - set_a
        
        recall = intersection / len_a if len_a > 0 else 1.0
        precision = intersection / len_b if len_b > 0 else 1.0
        iou = intersection / (len_a + len_b - intersection) if (len_a + len_b - intersection) > 0 else 1.0
        
        print(f"\n--- {label} Metrics ---")
        print(f"CPU Hits (A):      {len_a}")
        print(f"DPU Hits (B):      {len_b}")
        print(f"Intersection:      {intersection}")
        print(f"A Only (Missed):   {len(a_only)}")
        print(f"B Only (Extra):    {len(b_only)}")
        print(f"Recall (B vs A):   {recall:.4f}")
        print(f"Precision (B vs A):{precision:.4f}")
        print(f"IoU:               {iou:.4f}")
        
        if not args.no_diag_check and (len(a_only) > 0 or len(b_only) > 0):
            print(f"\n  [Deep Dive: {granularity} Validity Check]")
            
            a_valid, a_invalid = analyze_subset(a_only, "A Only", granularity, lookup_a, queries, targets, args.kmer, mask_offsets)
            print(f"  A Only (Missed): {len(a_only)}")
            print(f"    - True Hits (Valid):         {a_valid:<5} (DPU Missed Real Hit)")
            print(f"    - False Positives (Invalid): {a_invalid:<5} (CPU Hallucination)")
            
            b_valid, b_invalid = analyze_subset(b_only, "B Only", granularity, lookup_b, queries, targets, args.kmer, mask_offsets)
            print(f"  B Only (Extra):  {len(b_only)}")
            print(f"    - True Hits (Valid):         {b_valid:<5} (DPU Found New Hit)")
            print(f"    - False Positives (Invalid): {b_invalid:<5} (DPU Hallucination)")

    # 1. Run Metric Blocks
    run_analysis_block(cpu_qt, dpu_qt, cpu_lookup, dpu_lookup, "Sequence Granularity (Q-T)", "Sequence")
    run_analysis_block(cpu_qtd, dpu_qtd, cpu_lookup, dpu_lookup, "Diagonal Granularity (Q-T-D)", "Diagonal")

    # 2. Run Score Analysis (Pandas)
    analyze_scores_pandas(args.cpu_tsv, args.dpu_tsv)

if __name__ == "__main__":
    main()