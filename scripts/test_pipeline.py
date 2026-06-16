import os
os.environ['PYTHONUTF8'] = '1'

from PIL import Image
from src.preprocessing.pipeline import pretraiter_image
from kraken import rpred
from kraken.lib import models

print('Chargement du modele HTR...')
modele = models.load_any('models/McCATMuS_nfd_nofix_V1.mlmodel')
print('Modele charge !')

resultats = []
for i in range(10):
    chemin = f'data/test/raw/ligne_{i:03d}.jpg'
    image = pretraiter_image(chemin)
    print(f'Ligne {i} pretraitee : {image.size}')

print('Pipeline pretraitement OK !')
