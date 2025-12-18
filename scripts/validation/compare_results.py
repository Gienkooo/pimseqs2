import sys
import os
import pandas as pd


def load_results(filepath):
    """Loads results from a TSV file.

    Handles both headered and headerless TSVs. The loader will try to
    interpret columns named `query`, `target`, `score` (case-insensitive).
    If headers are missing, it will treat the first three columns as
    query, target, score. Additional columns such as `q_len`/`t_len` are
    preserved when present.
    """
    try:
        # First try: file has a header with named columns
        df = pd.read_csv(filepath, sep='\t', header=0)
        cols_lower = [c.lower() for c in df.columns]
        if 'query' in cols_lower and 'target' in cols_lower and 'score' in cols_lower:
            # normalize column names to lower-case
            df.columns = cols_lower
            return df

        # Fallback: no usable header -> read without header and assign defaults
        df = pd.read_csv(filepath, sep='\t', header=None)
        # Build sensible default column names for first columns
        colnames = []
        for i in range(df.shape[1]):
            if i == 0:
                colnames.append('query')
            elif i == 1:
                colnames.append('target')
            elif i == 2:
                colnames.append('score')
            elif i == 3:
                colnames.append('q_len')
            elif i == 4:
                colnames.append('t_len')
            else:
                colnames.append(f'col{i}')
        df.columns = colnames
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def compare(file_a, file_b, label):
    print(f"--- Comparison Report: {label} ---")
    a_df = load_results(file_a)
    b_df = load_results(file_b)

    name_a = os.path.basename(file_a)
    name_b = os.path.basename(file_b)

    print(f"File A ({name_a}) hits: {len(a_df)}")
    print(f"File B ({name_b}) hits: {len(b_df)}")

    # Create sets of (query, target) pairs. Ensure string types for consistency
    a_pairs = set(zip(a_df['query'].astype(str), a_df['target'].astype(str)))
    b_pairs = set(zip(b_df['query'].astype(str), b_df['target'].astype(str)))

    intersection = a_pairs.intersection(b_pairs)
    a_only = a_pairs - b_pairs
    b_only = b_pairs - a_pairs

    print(f"Intersection: {len(intersection)}")
    print(f"A only: {len(a_only)}")
    print(f"B only: {len(b_only)}")

    if len(a_pairs) > 0:
        recall = len(intersection) / len(a_pairs)
        print(f"Recall (B vs A): {recall:.4f}")

    if len(b_pairs) > 0:
        precision = len(intersection) / len(b_pairs)
        print(f"Precision (B vs A): {precision:.4f}")

    union_size = len(a_pairs.union(b_pairs))
    iou = len(intersection) / union_size if union_size > 0 else 0
    print(f"IoU: {iou:.4f}")

    # Compare scores for intersection
    if len(intersection) > 0:
        # Merge on query and target
        merged = pd.merge(a_df, b_df, on=['query', 'target'], suffixes=('_a', '_b'))

        # Attempt to coerce score columns to numeric
        merged['score_a'] = pd.to_numeric(merged['score_a'], errors='coerce')
        merged['score_b'] = pd.to_numeric(merged['score_b'], errors='coerce')

        merged['diff'] = merged['score_a'] - merged['score_b']

        corr = merged['score_a'].corr(merged['score_b'])
        print(f"Score correlation: {corr:.4f}" if pd.notna(corr) else "Score correlation: NA")
        print(f"Max score diff: {merged['diff'].abs().max()}" )
        print(f"Mean score diff: {merged['diff'].abs().mean():.4f}")

        # Check for significant differences and show largest diffs first
        significant_diff = merged[merged['diff'].abs() > 1.0]  # Tolerance of 1 bit/score
        if not significant_diff.empty:
            significant_diff = significant_diff.reindex(significant_diff['diff'].abs().sort_values(ascending=False).index)
            print(f"WARNING: {len(significant_diff)} hits have score difference > 1.0")
            cols = ['query', 'target', 'score_a', 'score_b', 'diff']
            if 'q_len' in significant_diff.columns and 't_len' in significant_diff.columns:
                cols = ['query', 'target', 'q_len', 't_len', 'score_a', 'score_b', 'diff']
            print(significant_diff[cols].head())

    print("-----------------------------------")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_results.py <fileA_tsv> <fileB_tsv> <label>")
        sys.exit(1)

    compare(sys.argv[1], sys.argv[2], sys.argv[3])
