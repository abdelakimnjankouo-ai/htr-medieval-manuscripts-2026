import os
from pathlib import Path
from PIL import Image

os.makedirs('data/train_xml', exist_ok=True)
os.makedirs('data/val_xml', exist_ok=True)

def creer_alto_xml(img_path, txt_path, out_path):
    img = Image.open(img_path)
    w, h = img.size
    with open(txt_path, encoding='utf-8') as f:
        texte = f.read().strip()
    texte = texte.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(chr(34),'&quot;')
    abs_img = str(Path(img_path).resolve()).replace(chr(92),'/')
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">\n  <Description>\n    <MeasurementUnit>pixel</MeasurementUnit>\n    <sourceImageInformation>\n      <fileName>' + abs_img + '</fileName>\n    </sourceImageInformation>\n  </Description>\n  <Layout>\n    <Page WIDTH="' + str(w) + '" HEIGHT="' + str(h) + '" ID="page1">\n      <PrintSpace>\n        <TextBlock ID="tb1">\n          <TextLine ID="tl1" BASELINE="' + str(h//2) + '">\n            <String CONTENT="' + texte + '" HPOS="0" VPOS="0" WIDTH="' + str(w) + '" HEIGHT="' + str(h) + '"/>\n          </TextLine>\n        </TextBlock>\n      </PrintSpace>\n    </Page>\n  </Layout>\n</alto>'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xml)

print('Generation XML train...')
count = 0
for i in range(300):
    img = f'data/train/ligne_{i:04d}.jpg'
    txt = f'data/train/ligne_{i:04d}.txt'
    out = f'data/train_xml/ligne_{i:04d}.xml'
    if Path(img).exists() and Path(txt).exists():
        creer_alto_xml(img, txt, out)
        count += 1
print(f'Train: {count} fichiers XML')

print('Generation XML val...')
count = 0
for i in range(50):
    img = f'data/val/ligne_{i:04d}.jpg'
    txt = f'data/val/ligne_{i:04d}.txt'
    out = f'data/val_xml/ligne_{i:04d}.xml'
    if Path(img).exists() and Path(txt).exists():
        creer_alto_xml(img, txt, out)
        count += 1
print(f'Val: {count} fichiers XML')
print('Termine !')
