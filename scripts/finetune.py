import os
os.environ['PYTHONUTF8'] = '1'

from pathlib import Path
from kraken.lib.train import GroundTruthDataset, KrakenTrainer
from kraken.lib import models

print('Preparation du dataset...')

# Lister les paires image/transcription
train_pairs = []
for i in range(300):
    img = Path(f'data/train/ligne_{i:04d}.jpg')
    txt = Path(f'data/train/ligne_{i:04d}.txt')
    if img.exists() and txt.exists():
        train_pairs.append((str(img), str(txt)))

val_pairs = []
for i in range(50):
    img = Path(f'data/val/ligne_{i:04d}.jpg')
    txt = Path(f'data/val/ligne_{i:04d}.txt')
    if img.exists() and txt.exists():
        val_pairs.append((str(img), str(txt)))

print(f'Train: {len(train_pairs)} paires')
print(f'Val: {len(val_pairs)} paires')

print('Lancement fine-tuning Kraken...')
os.system(f'ketos train -f alto -d cpu --lag 5 --min-epochs 2 --max-epochs 10 -r 0.0001 -m models/McCATMuS_nfd_nofix_V1.mlmodel -o models/finetuned ' + ' '.join([p[0] for p in train_pairs]))
