import os, sys
os.environ['PYTHONUTF8'] = '1'
sys.path.insert(0, '.')
from pathlib import Path
from kraken.lib import models

print('Chargement modele...')
m = models.load_any('models/McCATMuS_nfd_nofix_V1.mlmodel')
print('Modele OK')

pairs = []
for i in range(50):
    img = Path(f'data/train_gt/ligne_{i:04d}.jpg')
    txt = Path(f'data/train_gt/ligne_{i:04d}.gt.txt')
    if img.exists() and txt.exists():
        pairs.append((str(img), str(txt)))
print(f'Paires trouvees: {len(pairs)}')
