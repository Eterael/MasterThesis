#!/usr/bin/env python3
import os
import subprocess
import csv
import gzip
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def parse_fasta(fasta_file):
    """Parse FASTA with class and allele info"""
    records = {}
    opener = gzip.open if fasta_file.endswith('.gz') else open
    with opener(fasta_file, 'rt') as f:
        current = {'header': '', 'seq': ''}
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current['header']:
                    parts = current['header'].split('|')
                    records[current['header']] = {
                        'seq': current['seq'],
                        'class': int(parts[1].split('=')[1]),
                        'allele': parts[2].split('=')[1]
                    }
                current = {'header': line[1:], 'seq': ''}
            else:
                current['seq'] += line
        if current['header']:
            parts = current['header'].split('|')
            records[current['header']] = {
                'seq': current['seq'],
                'class': int(parts[1].split('=')[1]),
                'allele': parts[2].split('=')[1]
            }
    return records

def run_blast(fasta_file, db_name="peptide_db"):
    """Run BLAST and find best non-self hit per query"""
    best_hits = {}
    
    # BLAST command configured for short peptides
    blast_cmd = (
        f"blastp -db {db_name} -query {fasta_file} "
        f"-outfmt '6 qseqid sseqid pident evalue' "
        f"-task blastp-short -evalue 1000 -max_target_seqs 2 "
        f"-comp_based_stats 0"  # Disable composition stats for short seqs
    )
    
    proc = subprocess.Popen(blast_cmd, shell=True, stdout=subprocess.PIPE, text=True)
    
    for line in proc.stdout:
        qseqid, sseqid, pident, evalue = line.strip().split('\t')
        if qseqid == sseqid:
            continue  # Skip self-hits
        
        # Keep only the best hit (first non-self due to BLAST sorting)
        if qseqid not in best_hits:
            best_hits[qseqid] = {
                'target': sseqid,
                'pident': float(pident),
                'evalue': float(evalue)
            }
    
    return best_hits

def evaluate_predictions(records, best_hits):
    """Calculate AUC and confusion matrix"""
    y_true, y_score = [], []
    
    for qheader, hit in best_hits.items():
        y_true.append(records[qheader]['class'])
        
        # Prediction score based on E-value (lower E-value = stronger prediction)
        score = -math.log10(hit['evalue']) if hit['evalue'] > 0 else 100
        y_score.append(score if records[hit['target']]['class'] == 1 else -score)
    
    # AUC-ROC calculation
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Confusion matrix (using 0.5 as threshold)
    y_pred = [1 if s > 0 else 0 for s in y_score]
    cm = confusion_matrix(y_true, y_pred)
    
    return auc, cm, fpr, tpr

def save_results(records, best_hits, auc, cm, output_prefix="results"):
    """Save all results to files"""
    # Save best hits
    with open(f"{output_prefix}_best_hits.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['query_header', 'query_class', 'query_allele',
                        'best_hit_header', 'best_hit_class', 'best_hit_allele',
                        'pident', 'evalue', 'score'])
        
        for qheader, hit in best_hits.items():
            target = hit['target']
            score = -math.log10(hit['evalue']) if hit['evalue'] > 0 else 100
            writer.writerow([
                qheader,
                records[qheader]['class'],
                records[qheader]['allele'],
                target,
                records[target]['class'],
                records[target]['allele'],
                hit['pident'],
                hit['evalue'],
                score
            ])
    
    # Save evaluation metrics
    with open(f"{output_prefix}_metrics.txt", 'w') as f:
        f.write(f"AUC: {auc:.4f}\n\nConfusion Matrix:\n{cm}")
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f"{output_prefix}_roc.png")
    plt.close()

def main(fasta_file):
    # 1. Load data
    print("Loading sequences...")
    records = parse_fasta(fasta_file)
    
    # 2. Create BLAST DB
    print("Creating BLAST database...")
    subprocess.run(f"makeblastdb -in {fasta_file} -dbtype prot -out peptide_db",
                  shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 3. Find best non-self hits
    print("Finding best hits...")
    best_hits = run_blast(fasta_file)
    
    # 4. Evaluate
    print("Evaluating predictions...")
    auc, cm, fpr, tpr = evaluate_predictions(records, best_hits)
    
    # 5. Save results
    print("Saving results...")
    save_results(records, best_hits, auc, cm)
    
    print(f"\nResults saved to:")
    print(f"- results_best_hits.csv : Best hit for each peptide")
    print(f"- results_metrics.txt   : AUC and confusion matrix")
    print(f"- results_roc.png       : ROC curve plot")
    print(f"\nAUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    import sys
    import math
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <peptides.fasta[.gz]>")
        sys.exit(1)
    
    main(sys.argv[1])