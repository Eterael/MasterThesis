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


depictions_dir = '../data/28struc/'
outdir = '../data/mhcpep_seq/'
infile = '../data/mhc_appended_sequences.txt'

amino_acid_full_names = {
    'A': 'alanine', 'R': 'arginine', 'N': 'asparagine', 'D': 'aspartic_acid', 'C': 'cysteine', 
    'E': 'glutamic_acid', 'Q': 'glutamine', 'G': 'glycine', 'H': 'histidine', 'I': 'isoleucine', 
    'L': 'leucine', 'K': 'lysine', 'M': 'methionine', 'F': 'phenylalanine', 'P': 'proline',
    'S': 'serine', 'T': 'threonine', 'W': 'tryptophan', 'Y': 'tyrosine', 'V': 'valine',
    's': 'phospho_serine', 't': 'phospho_threonine', 'y': 'phospho_tyrosine'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

amino_acid_dict = {}
        
for letter, aa in amino_acid_full_names.items():
    img_path = f'{depictions_dir}/{aa}.png'
    image = Image.open(img_path).convert('RGB')
    amino_acid_dict[letter] = image

width, height = amino_acid_dict['A'].size


def concat_pictures(amino_acid_sequence, amino_acid_dict):
    
    new_img = Image.new('RGB', (len(amino_acid_sequence)*width, height))
    
    i = 0
    for aa in amino_acid_sequence:
        im = amino_acid_dict[aa]
        new_img.paste(im, (i,0))
        i += width
         
    return new_img

f = open(infile, 'r')
proteins = f.read().split('\n')
f.close()
prot_seq = [peptide for peptide in proteins if peptide]

for protein in prot_seq:
    img = concat_pictures(protein, amino_acid_dict)
    img.save(f'{outdir}/{protein}.png')

