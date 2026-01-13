#!/usr/bin/env python3
import argparse
import sys

RED = '\033[91m'
RESET = '\033[0m'

def parse_args():
    parser = argparse.ArgumentParser(description="Verify k-mer hits on diagonals.")
    parser.add_argument("--query", required=True, help="Path to Query FASTA file")
    parser.add_argument("--target", required=True, help="Path to Target FASTA file")
    parser.add_argument("--tsv", required=True, help="Path to TSV file")
    parser.add_argument("--kmer", type=int, default=7, help="K-mer size")
    parser.add_argument("--mask", type=str, default="1111111", help="Binary mask string")
    parser.add_argument("--visualize", action="store_true", help="Print aligned sequences with matches in red")
    parser.add_argument("--check-counts", action="store_true", help="Enable strict verification of exact hit counts.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed table of errors and summary.")
    return parser.parse_args()

def parse_fasta(filepath):
    seqs = {}
    name = None
    seq = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if name: seqs[name] = ''.join(seq)
                    raw_header = line[1:].split()[0]
                    if '|' in raw_header and raw_header.count('|') >= 2:
                        name = raw_header.split('|')[1]
                    else:
                        name = raw_header
                    seq = []
                else:
                    seq.append(line)
            if name: seqs[name] = ''.join(seq)
    except FileNotFoundError:
        if "--verbose" in sys.argv:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    return seqs

def apply_mask(sequence, mask):
    if len(sequence) != len(mask):
        return sequence # Fallback if lengths differ unpredictably
    return "".join([s for s, m in zip(sequence, mask) if m == '1'])

def check_diagonal_hits(q_seq, t_seq, diag, mask):
    window_size = len(mask)
    matches = []

    for i in range(len(q_seq) - window_size + 1):
        j = i - diag
        if j < 0 or j > len(t_seq) - window_size:
            continue
        q_window = q_seq[i : i + window_size]
        t_window = t_seq[j : j + window_size]
        if apply_mask(q_window, mask) == apply_mask(t_window, mask):
            matches.append((i, j, q_window))
    return matches

def main():
    args = parse_args()
    
    if args.verbose:
        print(f"Loading FASTA files...", file=sys.stderr)
        
    queries = parse_fasta(args.query)
    targets = parse_fasta(args.target)
    
    total_processed = 0
    wrong_diagonal_existence_count = 0 
    wrong_count_val_count = 0          
    
    error_lines = []

    try:
        with open(args.tsv, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4: continue
                
                raw_q_id, raw_t_id, claimed_hits_str, diag_str = parts[0], parts[1], parts[2], parts[3]

                q_id = raw_q_id.split('|')[1] if '|' in raw_q_id and raw_q_id.count('|')>=2 else raw_q_id
                t_id = raw_t_id.split('|')[1] if '|' in raw_t_id and raw_t_id.count('|')>=2 else raw_t_id
                
                if q_id not in queries: q_id = raw_q_id if raw_q_id in queries else None
                if t_id not in targets: t_id = raw_t_id if raw_t_id in targets else None

                if not q_id or not t_id: continue

                claimed_hits = int(claimed_hits_str)
                diag = int(diag_str)
                
                matches = check_diagonal_hits(queries[q_id], targets[t_id], diag, args.mask)
                verified_count = len(matches)
                
                status = "OK"
                
                # Case 1: False Positive 
                if claimed_hits > 0 and verified_count == 0:
                    status = "FALSE_POS_DIAG"
                    wrong_diagonal_existence_count += 1
                elif claimed_hits == 0 and verified_count > 0:
                    status = "FALSE_NEG_DIAG"
                    wrong_diagonal_existence_count += 1
                
                # Case 2: Count Mismatch (with 255 cap logic)
                elif args.check_counts:
                    if claimed_hits == 255 and verified_count >= 255:
                        status = "OK"
                    elif claimed_hits != verified_count:
                        status = "COUNT_MISMATCH"
                        wrong_count_val_count += 1
                
                if status != "OK":
                    if args.verbose:
                        error_line = f"{q_id:<15} {t_id:<15} {diag:<10} {claimed_hits:<10} {verified_count:<10} {status}"
                        error_lines.append(error_line)
                
                total_processed += 1

    except FileNotFoundError:
        if args.verbose:
            print(f"Error: TSV file not found: {args.tsv}", file=sys.stderr)
        sys.exit(1)

    fp_rate = 0.0
    if total_processed > 0:
        fp_rate = (wrong_diagonal_existence_count / total_processed) * 100

    if args.verbose:
        if error_lines:
            print(f"{'Q_ID':<15} {'T_ID':<15} {'DIAG':<10} {'CLAIMED':<10} {'VERIFIED':<10} {'STATUS'}")
            print("-" * 80)
            for line in error_lines:
                print(line)
            print("-" * 80)
        
        print("\n" + "="*40)
        print(f"SUMMARY FOR {args.tsv}")
        print("="*40)
        print(f"Total Lines Processed: {total_processed}")
        print(f"False Positives (Wrong Diagonal Existence): {wrong_diagonal_existence_count} ({fp_rate:.2f}%)")
        if args.check_counts:
            print(f"Double hit counts verified wrong (Count mismatch): {wrong_count_val_count}")
    else:
        print(f"{fp_rate:.2f}")

if __name__ == "__main__":
    main()