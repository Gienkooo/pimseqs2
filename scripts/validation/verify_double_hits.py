#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict

# ==============================================================================
# 1. SETUP & PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Schema Verification for MMseqs2 DPU K-mer Filter.")
    parser.add_argument("--query", required=True, help="Query FASTA")
    parser.add_argument("--target", required=True, help="Target FASTA")
    parser.add_argument("--tsv", required=True, help="Results TSV (Query Target Score Diag)")
    parser.add_argument("--kmer", type=int, default=7, help="K-mer size")
    parser.add_argument("--mask", type=str, default="1111111", help="Binary mask")
    parser.add_argument("--log", type=str, help="Path to write verbose logs. Summary is always printed to stdout.")
    return parser.parse_args()

def parse_fasta(filepath):
    seqs = {}
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
                    if '|' in raw_header and raw_header.count('|') >= 2:
                        name = raw_header.split('|')[1]
                    else:
                        name = raw_header
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if name: seqs[name] = ''.join(seq_parts)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    return seqs

def parse_tsv(filepath):
    data = defaultdict(list)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                q_raw, t_raw = parts[0], parts[1]
                q_id = q_raw.split('|')[1] if '|' in q_raw and q_raw.count('|') >= 2 else q_raw
                t_id = t_raw.split('|')[1] if '|' in t_raw and t_raw.count('|') >= 2 else t_raw
                
                try:
                    score = int(parts[2])
                    diag = int(parts[3]) if len(parts) > 3 else 0 
                except ValueError:
                    continue

                data[(q_id, t_id)].append({'score': score, 'diag': diag})
    except FileNotFoundError:
        return None
    return data

# ==============================================================================
# 2. LOGGING HELPERS
# ==============================================================================

class Logger:
    def __init__(self, log_file=None):
        self.file_handle = open(log_file, 'w') if log_file else None

    def verbose(self, msg):
        """Writes to log file if open, else stdout."""
        if self.file_handle:
            self.file_handle.write(msg + "\n")
        else:
            print(msg)

    def summary(self, msg):
        """Always writes to stdout (captured by bash variable), AND log file."""
        print(msg)
        if self.file_handle:
            self.file_handle.write(msg + "\n")
            
    def close(self):
        if self.file_handle:
            self.file_handle.close()

# ==============================================================================
# 3. CORE LOGIC
# ==============================================================================

def extract_masked_kmer(sequence, start_pos, mask_indices):
    chars = []
    try:
        for offset in mask_indices:
            chars.append(sequence[start_pos + offset])
        return "".join(chars)
    except IndexError:
        return None

def find_raw_hits(q_seq, t_seq, mask_str):
    hits = []
    mask_indices = [i for i, char in enumerate(mask_str) if char == '1']
    span = len(mask_str)
    
    target_index = defaultdict(list)
    for j in range(len(t_seq) - span + 1):
        kmer = extract_masked_kmer(t_seq, j, mask_indices)
        if kmer: target_index[kmer].append(j)
        
    for i in range(len(q_seq) - span + 1):
        kmer = extract_masked_kmer(q_seq, i, mask_indices)
        if kmer and kmer in target_index:
            for j in target_index[kmer]:
                hits.append((i, j, i - j))
    return hits

def simulate_double_hits(hits):
    if not hits: return 0
    hits.sort(key=lambda x: x[0])
    count = 0
    last_diag = -999999999
    for _, _, diag in hits:
        if diag == last_diag:
            count += 1
        last_diag = diag
    return count

def check_diagonal_existence(q_seq, t_seq, target_diag, mask_indices, span):
    start_i = max(0, target_diag)
    end_i = min(len(q_seq), len(t_seq) + target_diag) - span + 1
    
    for i in range(start_i, end_i):
        j = i - target_diag
        if j < 0 or j > len(t_seq) - span: continue
        
        match = True
        for offset in mask_indices:
            if q_seq[i + offset] != t_seq[j + offset]:
                match = False
                break
        if match: return True
    return False

def run_exact_simulation_consistency_check(dpu_data, queries, targets, mask_str, log):
    log.verbose("\nSimulation Consistency (Sum of Hits vs Python Sim)")
    log.verbose(f"{'QUERY':<15} {'TARGET':<15} {'DPU_SUM':<8} {'SIM_SUM':<8} {'STATUS'}")
    log.verbose("-" * 65)
    
    mismatches = 0
    processed = 0
    
    for (q_id, t_id), hits in dpu_data.items():
        if q_id not in queries or t_id not in targets: continue
        
        dpu_total = sum(h['score'] for h in hits)
        raw_hits = find_raw_hits(queries[q_id], targets[t_id], mask_str)
        sim_total = simulate_double_hits(raw_hits)
        
        processed += 1
        
        if dpu_total != sim_total:
            mismatches += 1
            log.verbose(f"{q_id:<15} {t_id:<15} {dpu_total:<8} {sim_total:<8} FAIL")
    
    acc = 100.0 * (1.0 - mismatches/processed) if processed else 0
    log.verbose("-" * 65)
    log.summary(f"Exact Simulation Accuracy: {acc:.2f}% ({processed-mismatches}/{processed}) unique sequence pairs")

def run_geometric_hit_validity_check(dpu_data, queries, targets, mask_str, log):
    log.verbose("\nGeometric Validity (Hit Existence on Diagonal)")
    log.verbose(f"{'QUERY':<15} {'TARGET':<15} {'DIAG':<8} {'SCORE':<6} {'STATUS'}")
    log.verbose("-" * 65)
    
    mask_indices = [i for i, char in enumerate(mask_str) if char == '1']
    span = len(mask_str)
    total_lines = 0
    false_positives = 0
    
    for (q_id, t_id), hits in dpu_data.items():
        if q_id not in queries or t_id not in targets: continue
        q_seq = queries[q_id]
        t_seq = targets[t_id]
        
        for h in hits:
            total_lines += 1
            diag = h['diag']
            exists = check_diagonal_existence(q_seq, t_seq, diag, mask_indices, span)
            if not exists:
                false_positives += 1
                log.verbose(f"{q_id:<15} {t_id:<15} {diag:<8} {h['score']:<6} INVALID")
                
    acc = 100.0 * (1.0 - false_positives/total_lines) if total_lines else 0
    log.verbose("-" * 65)
    log.summary(f"Geometric Hit Accuracy: {acc:.2f}% ({total_lines-false_positives}/{total_lines}) diagonal matches")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    log = Logger(args.log)
    
    print(f"Loading sequences...", file=sys.stderr)
    queries = parse_fasta(args.query)
    targets = parse_fasta(args.target)
    
    print(f"Loading TSV...", file=sys.stderr)
    dpu_data = parse_tsv(args.tsv)
    if not dpu_data:
        print("Failed to load data.")
        sys.exit(1)
        
    run_exact_simulation_consistency_check(dpu_data, queries, targets, args.mask, log)
    run_geometric_hit_validity_check(dpu_data, queries, targets, args.mask, log)
    
    log.close()

if __name__ == "__main__":
    main()