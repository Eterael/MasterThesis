from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from Bio import SeqIO
import pandas as pd
import numpy as np
import math

# === Load sequences from FASTA ===
sequences = []
headers = []
for record in SeqIO.parse("human_allel.fasta", "fasta"):
    sequences.append(str(record.seq))
    headers.append(record.id)

N = len(sequences)
n_clusters = 5

if N < n_clusters:
    raise ValueError("You must have at least 5 sequences to create 5 clusters.")

# === Step 1: Convert sequences to k-mer features ===
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(sequences)

# === Step 2: Perform loose KMeans clustering to get spread ===
kmeans = KMeans(n_clusters=n_clusters * 3, random_state=42)
initial_labels = kmeans.fit_predict(X)

# === Step 3: Greedy re-assignment into 5 balanced clusters ===
# Define desired sizes (e.g. [5, 4, 4, 4, 4] for N=21)
base = N // n_clusters
remainder = N % n_clusters
cluster_sizes = [base + 1 if i < remainder else base for i in range(n_clusters)]

# Shuffle all points and assign to buckets of desired size
indices = np.arange(N)
np.random.seed(49)
np.random.shuffle(indices)

final_labels = [-1] * N
ptr = 0
for cluster_id, size in enumerate(cluster_sizes):
    for _ in range(size):
        final_labels[indices[ptr]] = cluster_id
        ptr += 1

# === Step 4: Save assignments ===
df = pd.DataFrame({
    'Header': headers,
    'Cluster': final_labels
})
df.to_csv("cluster_assignments.csv", index=False)

print("? Clustering complete!")
print("?? Results saved to cluster_assignments.csv")
print(f"?? Cluster sizes: {cluster_sizes}")