from datasets import load_dataset
import os

ds = load_dataset('CATMuS/medieval', split='test', streaming=True)
os.makedirs('data/test/raw', exist_ok=True)
os.makedirs('data/test/transcription', exist_ok=True)

for i, sample in enumerate(ds):
    if i >= 10:
        break
    sample['im'].save(f'data/test/raw/ligne_{i:03d}.jpg')
    with open(f'data/test/transcription/ligne_{i:03d}.txt', 'w', encoding='utf-8') as f:
        f.write(sample['text'])
    print(f'Ligne {i}: ' + sample['text'][:50])

print('Termine !')
