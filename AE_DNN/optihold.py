import argparse
import torch
import numpy as np
import os
import glob
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score, roc_curve, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, required=True, help="cluster number")
parser.add_argument("--encoding", type=str, choices=["auto_encoder", "onehot"], default="auto_encoder")
args = parser.parse_args()

# === Data paths ===
depictions_dir = '../data/28struc/'
indir = '../data/28pep_seq/'

if args.fold == 't':
    train_file = f'../data/test/folds/fold_{args.fold}/train.txt'
    test_file = f'../data/test/folds/fold_{args.fold}/test.txt'
else:
    train_file = f'../data/folds_clusters/fold_{args.fold}/train.txt'
    test_file = f'../data/folds_clusters/fold_{args.fold}/test.txt'

# === Load test dataset ===
test_dataset = torch.load(f'folds/28tot_fold_{args.fold}_test.pt')

if args.encoding == 'auto_encoder':
    X_test = test_dataset['features']
    y_test = test_dataset['labels'].float()
elif args.encoding == 'onehot':
    def one_hot_encode_peptides(peptides):
        from sklearn.preprocessing import OneHotEncoder
        flattened_peptides = np.array(peptides).flatten().reshape(-1, 1)
        encoder = OneHotEncoder(categories=[list('ACDEFGHIKLMNPQRSTVWYstyh')], sparse_output=False)
        one_hot_encoded = encoder.fit_transform(flattened_peptides)
        num_peptides = len(peptides)
        return one_hot_encoded.reshape(num_peptides, -1)

    X_test = torch.tensor(one_hot_encode_peptides([list(p) for p in test_dataset['peptides']]), dtype=torch.float32)
    y_test = test_dataset['labels'].float()

input_size = X_test.shape[1]

# === Define model ===
class Linear_NN(torch.nn.Module):

    def __init__(self, input_size):
        super(Linear_NN, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)  # Optional, depending on your task (e.g., binary classification)
        return out

# === Ensemble prediction for current fold ===
n_splits = 4
device = torch.device('cpu')
ensemble_probs = []
inner_thresholds = []

for i in range(n_splits):
    model = Linear_NN(input_size).to(device)
    model.load_state_dict(torch.load(f'folds/results_cl/models/of_{args.fold}_{args.encoding}_model_fold_{i}.pt', map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(X_test).cpu().numpy().squeeze()
        ensemble_probs.append(preds)

    # === Compute optimal threshold where sensitivity = specificity for this inner fold ===
    fpr_i, tpr_i, thresholds_i = roc_curve(y_test.numpy(), preds)
    specificity_i = 1 - fpr_i
    diff_i = np.abs(tpr_i - specificity_i)
    best_idx_i = np.argmin(diff_i)
    best_threshold_i = thresholds_i[best_idx_i]
    inner_thresholds.append(best_threshold_i)

    # === Save inner fold threshold ===
    with open(f'optihold/of_{args.fold}_{args.encoding}_inner_threshold_fold_{i}.txt', 'w') as tf:
        tf.write(f"Optimal threshold (Sensitivity = Specificity) for inner fold {i}: {best_threshold_i:.4f}\n")

avg_probs = np.mean(ensemble_probs, axis=0)
y_true = y_test.numpy()

# === Compute optimal threshold where sensitivity = specificity for ensemble ===
fpr, tpr, thresholds = roc_curve(y_true, avg_probs)
specificity = 1 - fpr
diff = np.abs(tpr - specificity)
best_idx = np.argmin(diff)
best_threshold = thresholds[best_idx]

# === Save threshold ===
with open(f'optihold/of_{args.fold}_{args.encoding}_threshold.txt', 'w') as tf:
    tf.write(f"Optimal threshold (Sensitivity = Specificity) for fold {args.fold}: {best_threshold:.4f}\n")

# === Use threshold for this fold's predictions ===
y_pred = (avg_probs >= best_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)

metrics = {
    'accuracy': accuracy_score,
    'f1_score': f1_score,
    'roc_auc': roc_auc_score,
    'average_precision': average_precision_score,
    'recall': recall_score,
    'precision': precision_score,
}

# === Save results ===
with open(f'optihold/of_{args.fold}_{args.encoding}_final_evaluation.txt', 'w') as f:
    f.write(f"Optimal threshold (Sensitivity = Specificity): {best_threshold:.3f}\n")
    for name, func in metrics.items():
        score = func(y_true, avg_probs if name == 'roc_auc' else y_pred)
        f.write(f"{name}: {score:.4f}\n")

    tn, fp, fn, tp = cm.ravel()
    f.write("\nConfusion Matrix:\n")
    f.write(f"[[TN FP]\n [FN TP]]\n[[{tn} {fp}]\n [{fn} {tp}]]\n")

    # === Phosphorylated peptide analysis ===
    phospho_indices = [i for i, p in enumerate(test_dataset['peptides']) if any(c.islower() for c in p)]
    y_true_phospho = y_true[phospho_indices]
    y_pred_phospho = y_pred[phospho_indices]
    probs_phospho = avg_probs[phospho_indices]

    if len(y_true_phospho) > 0:
        phospho_cm = confusion_matrix(y_true_phospho, y_pred_phospho)
        if phospho_cm.shape == (2, 2):
            tn_p, fp_p, fn_p, tp_p = phospho_cm.ravel()
        else:
            tn_p = fp_p = fn_p = tp_p = 0
    else:
        tn_p = fp_p = fn_p = tp_p = 0
        phospho_cm = np.zeros((2, 2), dtype=int)
        f.write("\nNo phosphorylated peptides were present in this test set.\n")

    f.write("\nConfusion Matrix for Phosphorylated Peptides:\n")
    f.write(f"[[TN FP]\n [FN TP]]\n[[{tn_p} {fp_p}]\n [{fn_p} {tp_p}]]\n")
    f.write(f"Number of phosphorylated peptides: {len(phospho_indices)}\n")

    if len(y_true_phospho) > 0:
        f.write("\nPerformance on Phosphorylated Peptides:\n")
        for name, func in metrics.items():
            score = func(y_true_phospho, probs_phospho if name == 'roc_auc' else y_pred_phospho)
            f.write(f"{name}: {score:.4f}\n")

# === Save raw outputs for this fold to be used in future ===
torch.save({
    'probs': avg_probs,
    'labels': y_true,
    'peptides': test_dataset['peptides']
}, f'optihold/of_{args.fold}_{args.encoding}_raw_outputs.pt')
