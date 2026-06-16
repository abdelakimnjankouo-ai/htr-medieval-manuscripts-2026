from datasets import load_dataset
import os

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

print('Telechargement train...')
ds_train = load_dataset('CATMuS/medieval', split='train', streaming=True)
ds_val = load_dataset('CATMuS/medieval', split='test', streaming=True)

for i, sample in enumerate(ds_train):
    if i >= 300: break
    sample['im'].save(f'data/train/ligne_{i:04d}.jpg')
    with open(f'data/train/ligne_{i:04d}.txt', 'w', encoding='utf-8') as f:
        f.write(sample['text'])
    if i % 50 == 0:
        print(f'Train: {i}/300')

print('Telechargement val...')
for i, sample in enumerate(ds_val):
    if i >= 50: break
    sample['im'].save(f'data/val/ligne_{i:04d}.jpg')
    with open(f'data/val/ligne_{i:04d}.txt', 'w', encoding='utf-8') as f:
        f.write(sample['text'])
    if i % 10 == 0:
        print(f'Val: {i}/50')

print('Termine !')
