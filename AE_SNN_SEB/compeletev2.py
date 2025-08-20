import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import os
import random
import argparse
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score, roc_curve, auc, confusion_matrix

import tqdm.notebook as tqdm
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, required=True, help="cluster number")
parser.add_argument("--test", action="store_true", help= "activate test mode with corresponding data")
args = parser.parse_args()

depictions_dir = '../data/28struc/'

if args.test:
    train_file = f'../data/test/folds/fold_{args.fold}/train.txt'
    test_file  = f'../data/test/folds/fold_{args.fold}/test.txt'
else:
    train_file = f'../data/folds_clusters/fold_{args.fold}/train.txt'
    test_file  = f'../data/folds_clusters/fold_{args.fold}/test.txt'

indir = '../data/28pep_seq/'
out = f"folds/28tot_cluster_{args.fold}.pt"
batch_size = 128

amino_acid_full_names = {
    'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartic_acid', 'C': 'cysteine',
    'E': 'glutamic_acid', 'Q': 'glutamine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine',
    'L': 'leucine', 'K': 'lysine', 'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline',
    'S': 'serine', 'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine',
    's': 'phospho_serine', 't': 'phospho_threonine', 'y': 'phospho_tyrosine'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amino_acid_dict = {}

class ProteinImageDataset(Dataset):
    def __init__(self, seq_list, indir, transform=None):
        self.seq_list = seq_list
        self.indir = indir
        self.transform = transform
    def __len__(self):
        return len(self.seq_list)
    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        img_path = f"{self.indir}{seq}.png"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, seq

df = pd.read_csv(train_file, sep=" ", header=None, names=["seq", "label", "origin"])
prot_seq = df["seq"].tolist()
del df

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(49)

transform = transforms.ToTensor()
full_dataset = ProteinImageDataset(prot_seq, indir, transform=transform)

print(f'number of training samples: {len(full_dataset)}')
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

RANDOM_SEED = 49
LEARNING_RATE = 0.0005
NUM_EPOCHS = 10

print('Training Set:\n')
for images, _ in train_loader:
    print('Image batch dimensions:', images.size())
    break

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, target_height, target_width):
        super(Trim, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
    def forward(self, x):
        return x[:, :, :self.target_height, :self.target_width]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AutoEncoderV2(nn.Module):
    def __init__(self):
        super(AutoEncoderV2, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(128),

            nn.Flatten(),
            nn.Linear(128 * 1 * 15, 256)  # Bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 1 * 15),
            Reshape(-1, 128, 1, 15),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),

            nn.Upsample(size=(28, 252), mode='bilinear', align_corners=False),
            Trim(28, 252)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoderV2()
x = torch.randn(1, 3, 28, 28*9)
output = model(x)
print(output.shape)

model = AutoEncoderV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_epoch_loss_autoencoder(model, dataloader, loss_fn):
    loss = 0.0
    predictions = []
    for features in dataloader:
        features = features.to(device)
        logits = model(features)
        loss += loss_fn(logits, features)
        predictions.extend(logits)
    return loss / len(dataloader), predictions

def train_model(num_epochs, model, optimizer, train_loader, loss_fn=None,
                logging_interval=100, skip_epoch_stats=False, save_model=None):
    log_dict = {'train_loss_per_batch': [], 'train_loss_per_epoch': []}
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):
            features = features.to(device)
            logits = model(features)
            print(logits.shape)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx, len(train_loader), loss))
        if not skip_epoch_stats:
            model.eval()
            with torch.no_grad():
                train_loss = compute_epoch_loss_autoencoder(model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    return log_dict

log_dict = train_model(num_epochs=NUM_EPOCHS, model=model,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       skip_epoch_stats=True,
                       logging_interval=250,
                       save_model=out)

class ProteinImageDataset(Dataset):
    def __init__(self, seq_list, indir, transform=None, label_map=None):
        self.seq_list = seq_list
        self.indir = indir
        self.transform = transform
        self.label_map = label_map
    def __len__(self):
        return len(self.seq_list)
    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        img_path = os.path.join(self.indir, f"{seq}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[seq] if self.label_map else 0
        return image, label, seq

df = pd.read_csv(train_file, sep=" ", header=None, names=["seq", "label", "origin"])
df["label"] = df["label"].astype(int)
peptides = df["seq"].tolist()
label_map = dict(zip(df["seq"], df["label"]))
full_dataset = ProteinImageDataset(peptides, indir, transform=transform, label_map=label_map)

model.to("cpu")
model.eval()
encoder = model.encoder

def extract_features(dataset, encoder):
    features, labels, names = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for imgs, lbls, ids in loader:
            encoded = encoder(imgs)
            features.append(encoded)
            labels.append(lbls)
            names.extend(ids)
    return torch.cat(features), torch.cat(labels), names

X_train, y_train, names_train = extract_features(full_dataset, encoder)
torch.save({'features': X_train, 'labels': y_train, 'peptides': names_train}, f'folds/28tot_fold_{args.fold}_train.pt')
del X_train, y_train, names_train

df = pd.read_csv(test_file, sep=" ", header=None, names=["seq", "label", "origin"])
df["label"] = df["label"].astype(int)
peptides = df["seq"].tolist()
label_map = dict(zip(df["seq"], df["label"]))
full_dataset = ProteinImageDataset(peptides, indir, transform=transform, label_map=label_map)
X_test, y_test, names_test = extract_features(full_dataset, encoder)
torch.save({'features': X_test, 'labels': y_test, 'peptides': names_test}, f'folds/28tot_fold_{args.fold}_test.pt')
del X_test, y_test, names_test

class PCAPeptideMNIST(Dataset):
    def __init__(self, file_path):
        data = torch.load(file_path)
        self.features = data['features']
        self.labels = data['labels']
        self.peptides = data['peptides']
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train = f'folds/28tot_fold_{args.fold}_train.pt'
test  = f'folds/28tot_fold_{args.fold}_test.pt'

train_dataset = torch.load(train)
test_dataset  = torch.load(test)
N_SPLITS = 4
N_EPOCHS  = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(49)

class Linear_NN(torch.nn.Module):
    def __init__(self, input_size):
        super(Linear_NN, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

def one_hot_encode_peptides(peptides):
    flattened_peptides = np.array(peptides).flatten().reshape(-1, 1)
    encoder = OneHotEncoder(categories=[list('ACDEFGHIKLMNPQRSTVWYstyh')], sparse_output=False)
    one_hot_encoded = encoder.fit_transform(flattened_peptides)
    num_peptides = len(peptides)
    one_hot_encoded = one_hot_encoded.reshape(num_peptides, -1)
    return one_hot_encoded

encoding_method = 'onehot'
if encoding_method == 'onehot':
    X_train = torch.tensor(one_hot_encode_peptides([list(peptide) for peptide in train_dataset['peptides']]), dtype=torch.float32).to(device)
    y_train = train_dataset['labels'].to(device)
    X_test  = torch.tensor(one_hot_encode_peptides([list(peptide) for peptide in test_dataset['peptides']]), dtype=torch.float32).to(device)
    y_test  = test_dataset['labels'].to(device)

input_size = X_train.shape[1]
y_train = y_train.float()
y_test  = test_dataset['labels'].to(device).float()

def plot_loss(train_loss, val_loss, val_acc, fold):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i, ytitle in enumerate(['loss', ['log loss']]):
        ax[i].plot(train_loss, label='train loss', linestyle='-.')
        ax[i].plot(val_loss, label='val loss', linestyle='-.')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(ytitle)
        ax[i].legend()
    ax[1].set_yscale('log')
    ax[2].plot(val_acc, label='val acc',  linestyle='-.')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()
    plt.savefig(f'folds/results_cl/of_{args.fold}_{encoding_method}_loss_fold_{fold}.png')
    plt.close()

def model_train(model, X_train, y_train, X_val, y_val, fold):
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = N_EPOCHS
    batch_size = 10
    batch_start = torch.arange(0, len(X_train), batch_size)
    train_loss, val_loss, val_acc = [], [], []
    best_acc = - np.inf
    best_weights = None
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True, leave=False) as bar:
            bar.set_description(f"Epoch {epoch+1}")
            for start in bar:
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                y_pred  = model(X_batch).view(-1)
                loss    = loss_fn(y_pred, y_batch)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))
        train_loss.append(epoch_loss/len(batch_start))
        model.eval()
        y_pred = model(X_val.to(device)).view(-1)
        epoch_val_loss = loss_fn(y_pred, y_val.to(device)).item()
        acc = accuracy_score(y_val.cpu().numpy(), y_pred.round().cpu().detach().numpy())
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        val_loss.append(epoch_val_loss)
        val_acc.append(acc)
    torch.save(best_weights, f'folds/results_cl/models/of_{args.fold}_{encoding_method}_model_fold_{fold}.pt')
    plot_loss(train_loss, val_loss, val_acc, fold)
    return best_acc

if args.fold == 't':
    outer_cluster = 0
else:
    outer_cluster = int(args.fold)

peptide_to_index = {pep: i for i, pep in enumerate(train_dataset['peptides'])}
all_cluster_files = [f'../data/clusters/cluster_{i}.txt' for i in range(5)]
inner_cluster_files = [f for i, f in enumerate(all_cluster_files) if i != outer_cluster]

cv_scores = []
all_preds = []
all_labels = []
all_peptides = []

for fold, val_cluster_file in enumerate(inner_cluster_files):
    with open(val_cluster_file, 'r') as f:
        val_peptides = [line.strip().split()[0] for line in f if line.strip()]
    val_indices = [peptide_to_index[p] for p in val_peptides if p in peptide_to_index]
    train_indices = [
        i for i in range(len(train_dataset['peptides']))
        if i not in val_indices and train_dataset['peptides'][i] not in val_peptides
    ]
    X_tr, y_tr = X_train[train_indices], y_train[train_indices]
    X_val, y_val = X_train[val_indices], y_train[val_indices]
    perm = torch.randperm(X_tr.size(0))
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]
    model = Linear_NN(input_size).to(device)
    acc = model_train(model, X_tr, y_tr, X_val, y_val, fold)
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_val).squeeze().cpu().numpy()
        y_true = y_val.cpu().numpy()
    all_preds.append(y_pred_prob)
    all_labels.append(y_true)
    peptides = np.array(train_dataset['peptides'])
    peptides_val = peptides[val_indices]
    all_peptides.append(peptides_val)
    df = pd.DataFrame({'peptide': peptides_val, 'label': y_true, 'prediction': y_pred_prob})
    df.to_csv(f'folds/results_cl/models/of_{args.fold}_{encoding_method}_fold_{fold}_results.csv', index=False)
    print(f"Accuracy {fold}: %.5f" % acc)
    cv_scores.append(acc)

# ---------- FIX: robust ROC helper (probabilities in; flatten) ----------
def plot_roc(y_true, y_pred, label=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')

# ---------- FIX: zoomed standardized pAUC@0.1 with correct random baseline ----------
from sklearn.metrics import roc_curve as _roc_curve, roc_auc_score as _roc_auc_score
def plot_zoomed_pauc01(y_true, y_scores, out_png, a=0.1, label='Ensemble'):
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()
    pauc = _roc_auc_score(y_true, y_scores, max_fpr=a)  # standardized pAUC in [0,1]
    fpr, tpr, _ = _roc_curve(y_true, y_scores)
    mask = fpr <= a
    plt.figure()
    plt.plot(fpr[mask], tpr[mask], label=f'{label} (AUC@{a} = {pauc:.3f})')
    # True random diagonal in the zoomed window
    plt.plot([0, a], [0, a], linestyle='--', color='gray', label=f'Random (AUC@{a} = 0.500)')
    plt.title(f"Zoomed ROC (FPR = {a})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0, a])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------- FIX: evaluate_ensembl rewritten (no Monte Carlo baseline) ----------
def evaluate_ensembl(X_test, y_test, n_splits):  
    perf_path = f'folds/results_cl/models/of_{args.fold}_{encoding_method}_performance.txt'
    with open(perf_path, 'w') as perf_f:

        metrics = {
            'accuracy': accuracy_score,
            'average precision score': average_precision_score,
            'recall score': recall_score,
            'f1 score': f1_score
        }

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random', color='k')

        ensembl_predictions = []
        score_lists = {name: [] for name in metrics.keys()}
        threshold = 0.35  # kept only for label-based metrics
        y_test_np = y_test.cpu().numpy().ravel()

        for i in range(n_splits):
            model = Linear_NN(input_size).to(device)
            model.load_state_dict(torch.load(
                f'folds/results_cl/models/of_{args.fold}_{encoding_method}_model_fold_{i}.pt',
                map_location=torch.device('cpu')))
            model.eval()
            with torch.no_grad():
                y_prob = model(X_test).detach().cpu().numpy().ravel()
            ensembl_predictions.append(torch.from_numpy(y_prob))

            # ROC per fold
            plot_roc(y_test_np, y_prob, label=f'Fold {i}')

            # per-fold label-based metrics at fixed threshold
            y_pred = (y_prob > threshold).astype(int)
            for name, scoring in metrics.items():
                score_lists[name].append(scoring(y_test_np, y_pred))

        avg_probs = torch.stack(ensembl_predictions).mean(dim=0).numpy().ravel()

        # CV summary
        print(f"{'-'*50}\nCross Validation\n{'-'*50}\n", file=perf_f)
        for name, scores in score_lists.items():
            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))
            print(f"{name}", file=perf_f)
            print(f"5-Fold Cross-Validation {name} scores: {np.round(scores, 3)}", file=perf_f)
            print(f"Mean {name} score: {mean_score:.3f} +/- {std_score:.3f}\n", file=perf_f)

        # Full ROC for ensemble probabilities
        plot_roc(y_test_np, avg_probs, label='Ensemble')
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.savefig(f'folds/results_cl/models/of_{args.fold}_{encoding_method}_roc.png')
        plt.close()

        # Ensemble label-based metrics at the same threshold
        y_pred_labels = (avg_probs > threshold).astype(int)
        print(f"{'-'*50}\nEnsembl\n{'-'*50}\n", file=perf_f)
        for name, scoring in metrics.items():
            print(f"{name}", file=perf_f)
            print(f"Ensembl {name} score: {scoring(y_test_np, y_pred_labels):.3f}\n", file=perf_f)

    # Save prediction table (labels + binarized per-fold preds + ensemble binarized)
    with open(f'folds/results_cl/models/of_{args.fold}_{encoding_method}_predictions.txt', 'w') as pred_f:
        print('ytest\tfold_1\tfold_2\tfold_3\tfold_4\tfold_5\tensembl', file=pred_f)
        fold_bin_preds = [(p.numpy().ravel() > threshold).astype(int) for p in ensembl_predictions]
        for i in range(len(y_test_np)):
            row = [str(y_test_np[i])]
            for fb in fold_bin_preds:
                row.append(str(fb[i]))
            row.append(str(y_pred_labels[i]))
            print('\t'.join(row), file=pred_f)

    # Zoomed standardized pAUC@0.1 with proper diagonal
    plot_zoomed_pauc01(
        y_test_np,
        avg_probs,
        f'folds/results_cl/models/of_{args.fold}_{encoding_method}_roc_zoomed.png',
        a=0.1,
        label='Ensemble'
    )

# ---------- FIX: stacked evaluation uses standardized pAUC@0.1 ----------
def evaluate_stacked_predictions(all_preds, all_labels, all_peptides, encoding_method="autoencoder"):
    y_all = np.concatenate(all_labels).ravel()
    y_pred_all = np.concatenate(all_preds).ravel()
    peptides_all = np.concatenate(all_peptides)

    fpr, tpr, _ = roc_curve(y_all, y_pred_all)
    auc_score = auc(fpr, tpr)

    pauc01 = roc_auc_score(y_all, y_pred_all, max_fpr=0.1)  # standardized partial AUC

    threshold = 0.35  # only for label metrics here
    y_pred_labels = (y_pred_all > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,   # full AUC here
        "average_precision": average_precision_score,
        "recall": recall_score,
        "precision": precision_score
    }

    results_file = f"folds/results_cl/models/of_{args.fold}_{encoding_method}_ensemble_scores.txt"
    with open(results_file, "w") as f:
        for name, metric in metrics.items():
            score = metric(y_all, y_pred_all) if name == "roc_auc" else metric(y_all, y_pred_labels)
            f.write(f"{name}: {score:.4f}\n")

        cm = confusion_matrix(y_all, y_pred_labels)
        tn, fp, fn, tp = cm.ravel()
        f.write("\nConfusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
        f.write("Matrix format:\n[[TN FP]\n [FN TP]]\n")
        f.write(f"{cm}\n")

        positive_rate = y_pred_labels.mean()
        f.write(f"\nPositive prediction rate: {positive_rate:.4f}\n")
        f.write(f"\nPartial AUC@0.1 (sklearn, standardized): {pauc01:.4f}\n")

    plt.figure()
    plt.plot(fpr, tpr, label=f"Ensemble ROC (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Ensemble ROC Curve")
    plt.legend()
    plt.savefig(f"folds/results_cl/models/of_{args.fold}_{encoding_method}_ensemble_roc.png")
    plt.close()

    df_all = pd.DataFrame({'peptide': peptides_all, 'label': y_all, 'prediction': y_pred_all})
    df_all.to_csv(f"folds/results_cl/models/of_{args.fold}_{encoding_method}_ensemble_results.csv", index=False)
    np.save(f"folds/results_cl/models/of_{args.fold}_{encoding_method}_ensemble_peptides.npy", peptides_all)

# ---- Run for one-hot ----
evaluate_ensembl(X_test, y_test, N_SPLITS)
evaluate_stacked_predictions(all_preds, all_labels, all_peptides, encoding_method=encoding_method)

# ---- Switch to autoencoder features ----
encoding_method = 'auto_encoder'
if encoding_method == 'auto_encoder':
    X_train = train_dataset['features'].to(device)
    y_train = train_dataset['labels'].to(device)
    X_test  = test_dataset['features'].to(device)
    y_test  = test_dataset['labels'].to(device)

input_size = X_train.shape[1]
y_train = y_train.float()
y_test  = test_dataset['labels'].to(device).float()

if args.fold == 't':
    outer_cluster = 0
else:
    outer_cluster = int(args.fold)

peptide_to_index = {pep: i for i, pep in enumerate(train_dataset['peptides'])}
all_cluster_files = [f'../data/clusters/cluster_{i}.txt' for i in range(5)]
inner_cluster_files = [f for i, f in enumerate(all_cluster_files) if i != outer_cluster]

cv_scores = []
all_preds = []
all_labels = []
all_peptides = []

for fold, val_cluster_file in enumerate(inner_cluster_files):
    with open(val_cluster_file, 'r') as f:
        val_peptides = [line.strip().split()[0] for line in f if line.strip()]
    val_indices = [peptide_to_index[p] for p in val_peptides if p in peptide_to_index]
    train_indices = [
        i for i in range(len(train_dataset['peptides']))
        if i not in val_indices and train_dataset['peptides'][i] not in val_peptides
    ]
    X_tr, y_tr = X_train[train_indices], y_train[train_indices]
    X_val, y_val = X_train[val_indices], y_train[val_indices]
    perm = torch.randperm(X_tr.size(0))
    X_tr = X_tr[perm]
    y_tr = y_tr[perm]
    model = Linear_NN(input_size).to(device)
    acc = model_train(model, X_tr, y_tr, X_val, y_val, fold)
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_val).squeeze().cpu().numpy()
        y_true = y_val.cpu().numpy()
    all_preds.append(y_pred_prob)
    all_labels.append(y_true)
    peptides = np.array(train_dataset['peptides'])
    peptides_val = peptides[val_indices]
    all_peptides.append(peptides_val)
    df = pd.DataFrame({'peptide': peptides_val, 'label': y_true, 'prediction': y_pred_prob})
    df.to_csv(f'folds/results_cl/models/of_{args.fold}_{encoding_method}_fold_{fold}_results.csv', index=False)
    print(f"Accuracy {fold}: %.5f" % acc)
    cv_scores.append(acc)

evaluate_ensembl(X_test, y_test, N_SPLITS)
evaluate_stacked_predictions(all_preds, all_labels, all_peptides, encoding_method=encoding_method)
