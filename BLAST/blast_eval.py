import os
import pandas as pd
import subprocess
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score
import numpy as np

# --- Configurable paths ---
data_dir = "../../data/folds_clusters"
#data_dir = "../../data/test/folds/
output_dir = "./blast_outputs"
db_path = "./blast_db/human_peptides"
blast_exec = "blastp"  # assumes it's in $PATH

os.makedirs(output_dir, exist_ok=True)

all_preds = []
all_labels = []

# --- Loop over 5 folds ---
for fold in range(5):
    print(f"Processing fold {fold}...")
    train_file = os.path.join(data_dir, f"fold_{fold}/train.txt")
    test_file = os.path.join(data_dir, f"fold_{fold}/test.txt")

    train_df = pd.read_csv(train_file, sep=" ", header=None, names=["seq", "label", "origin"])
    test_df = pd.read_csv(test_file, sep=" ", header=None, names=["seq", "label", "origin"])

    # --- Skip phosphorylated sequences (lowercase letters) ---
    train_df = train_df[~train_df['seq'].str.contains(r'[a-z]')]
    test_df = test_df[~test_df['seq'].str.contains(r'[a-z]')]

    # --- Write training FASTA ---
    train_fasta = os.path.join(output_dir, f"fold_{fold}_train.fasta")
    with open(train_fasta, 'w') as f:
        for i, row in train_df.iterrows():
            f.write(f">{row['origin']}\n{row['seq']}\n")

    # --- Make BLAST DB from training peptides ---
    subprocess.run(["makeblastdb", "-in", train_fasta, "-dbtype", "prot", "-out", db_path], check=True)

    # --- Write test FASTA ---
    test_fasta = os.path.join(output_dir, f"fold_{fold}_test.fasta")
    with open(test_fasta, 'w') as f:
        for i, row in test_df.iterrows():
            f.write(f">{row['origin']}\n{row['seq']}\n")

    # --- Run BLAST ---
    blast_out = os.path.join(output_dir, f"fold_{fold}_blast.tsv")
    subprocess.run([
        blast_exec,
        "-query", test_fasta,
        "-db", db_path,
        "-outfmt", "6 qseqid sseqid pident evalue bitscore",
        "-max_target_seqs", "1",
        "-evalue", "10",
        "-out", blast_out
    ], check=True)

    # --- Process BLAST output ---
    blast_df = pd.read_csv(blast_out, sep="\t", header=None,
                           names=["qseqid", "sseqid", "pident", "evalue", "bitscore"])
    blast_df = blast_df.drop_duplicates(subset='qseqid', keep='first')

    # Join predictions to labels
    merged = test_df.set_index("origin").join(
        blast_df.set_index("qseqid"), how="left"
    )
    merged['pident'] = merged['pident'].fillna(0)
    merged['prediction'] = merged['pident'] / 100  # Normalize to [0, 1]

    y_true = merged['label'].values
    y_pred = merged['prediction'].values
    y_pred_bin = (y_pred > 0.5).astype(int)

    all_preds.append(y_pred)
    all_labels.append(y_true)

    # --- Save predictions for ensemble later ---
    merged[['seq', 'label', 'prediction']].to_csv(
        os.path.join(output_dir, f"fold_{fold}_blast_predictions.csv"), index=False
    )

    # --- Evaluation ---
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    max_fpr = 0.1
    mask = fpr <= max_fpr
    partial_auc = np.trapz(tpr[mask], fpr[mask]) / max_fpr

    cm = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()

    with open(os.path.join(output_dir, f"fold_{fold}_blast_metrics.txt"), 'w') as f:
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write(f"Partial AUC@0.1: {partial_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
        f.write(f"[[{tn} {fp}]\n [{fn} {tp}]]\n")
        f.write(f"\nPositive Prediction Rate: {y_pred_bin.mean():.4f}\n")

    print(f"? Finished fold {fold}: AUC={auc_score:.4f}, pAUC@0.1={partial_auc:.4f}")

# --- Ensemble Evaluation ---
print("?? Evaluating ensemble performance...")
y_all = np.concatenate(all_labels)
y_pred_all = np.concatenate(all_preds)
y_pred_bin = (y_pred_all > 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_all, y_pred_all)
auc_score = roc_auc_score(y_all, y_pred_all)

max_fpr = 0.1
mask = fpr <= max_fpr
partial_auc = np.trapz(tpr[mask], fpr[mask]) / max_fpr

cm = confusion_matrix(y_all, y_pred_bin)
tn, fp, fn, tp = cm.ravel()

with open(os.path.join(output_dir, f"ensemble_blast_metrics.txt"), 'w') as f:
    f.write(f"[ENSEMBLE BLAST]\n")
    f.write(f"AUC: {auc_score:.4f}\n")
    f.write(f"Partial AUC@0.1: {partial_auc:.4f}\n")
    f.write(f"\nConfusion Matrix:\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
    f.write(f"[[{tn} {fp}]\n [{fn} {tp}]]\n")
    f.write(f"\nPositive Prediction Rate: {y_pred_bin.mean():.4f}\n")

print(f"?? Ensemble AUC={auc_score:.4f}, pAUC@0.1={partial_auc:.4f}")
