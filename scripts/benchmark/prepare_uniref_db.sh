#!/usr/bin/env bash

TARGET_SIZE=1000000         # 1 Million sequences for Target
QUERY_SIZE=10000

wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip uniref50.fasta.gz

# Linearize (remove line breaks in sequences) and Shuffle
# We use a temp file to avoid holding everything in RAM if using 'shuf' on massive files directly
awk '/^>/ {printf("\n%s\t",$0);next;} {printf("%s",$0);} END {printf("\n");}' "uniref50.fasta" \
| grep -v "^$" \
| head -n $((TARGET_SIZE + QUERY_SIZE)) \
| shuf > "uniref50_randomized.tsv"

echo "[+] Creating Target DB data ($TARGET_SIZE sequences)..."
head -n "$TARGET_SIZE" "uniref50_randomized.tsv" | tr "\t" "\n" > "target.fasta"

echo "[+] Creating Query DB data ($QUERY_SIZE sequences)..."
# We take queries from the lines AFTER the target set to avoid self-hits of the exact same index
tail -n +$((TARGET_SIZE + 1)) "uniref50_randomized.tsv" | head -n "$QUERY_SIZE" | tr "\t" "\n" > "query.fasta"
