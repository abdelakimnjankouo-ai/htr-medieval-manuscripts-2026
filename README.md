# HTR Medieval Manuscripts 2026

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Tests](https://img.shields.io/badge/tests-32%2F32%20passing-brightgreen.svg)]()
[![CER Baseline](https://img.shields.io/badge/CER%20baseline-32.5%25-orange.svg)]()

Pipeline complet de Reconnaissance Automatique de Texte Manuscrit (HTR) appliqué aux manuscrits médiévaux français — Projet MD5 HETIC 2026.

## Présentation

Ce projet implémente un pipeline bout-en-bout pour la transcription automatique de manuscrits médiévaux français (XIIIe–XVe siècles). Il compare deux architectures HTR : McCATMuS (CRNN+CTC) et un modèle Transformer, sur le corpus CATMuS Medieval.

**Résultat principal :** CER baseline = 32.5% (McCATMuS sans fine-tuning sur corpus CATMuS Medieval).

## Pipeline

```
Scan Gallica/BnF
      ↓
Prétraitement (CLAHE + Sauvola + deskew)
      ↓
Segmentation layout (SAM zero-shot)
      ↓
Segmentation lignes (Kraken BLLA — baselines)
      ↓
HTR : McCATMuS (CRNN+CTC) + Transformer
      ↓
Vote pondéré ensemble (Needleman-Wunsch)
      ↓
Export JSON (data contract)
```

## Structure du projet

```
htr-medieval-manuscripts-2026/
├── src/
│   ├── preprocessing/pipeline.py     # CLAHE + Sauvola + deskew
│   ├── segmentation/lines.py         # Kraken BLLA + IoU
│   ├── htr/kraken_htr.py             # Reconnaissance HTR
│   ├── evaluation/metrics.py         # CER/WER + bootstrap + McNemar
│   └── utils/seeds.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   └── test_data_contract.py         # 32/32 tests passent
├── scripts/
│   ├── download_data.py              # Téléchargement CATMuS
│   ├── prepare_corpus.py
│   └── train_kraken.py
├── data/
│   ├── train/                        # 300 images lignes
│   ├── val/                          # 50 images lignes
│   └── test/                         # 10 images + transcriptions
├── models/
│   ├── McCATMuS_nfd_nofix_V1.mlmodel
│   └── ManuMcFondue.mlmodel
├── article.md                        # Article scientifique
├── CONVENTIONS_TRANSCRIPTION.md
├── MODEL_CARD.md
├── DATA_SOURCES.md
└── requirements.txt
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/abdelakimnjankouo-ai/htr-medieval-manuscripts-2026.git
cd htr-medieval-manuscripts-2026

# Créer l'environnement virtuel (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# ou : source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation rapide

```python
from src.preprocessing.pipeline import pretraiter_image
from src.evaluation.metrics import calculer_cer

# Prétraiter une image
image_traitee = pretraiter_image("data/test/raw/image.jpg")

# Calculer le CER
cer = calculer_cer(hypothese="transcription modele", reference="transcription reference")
print(f"CER : {cer:.1%}")
```

## Lancer les tests

```bash
pytest tests/ -v
# 32/32 tests passent
```

## Résultats

| Modèle | Architecture | CER | Notes |
|--------|-------------|-----|-------|
| McCATMuS | CRNN+CTC | 32.5% | Sans fine-tuning |
| ManuMcFondue | CRNN+CTC | — | Modèle alternatif |
| Transformer | TrOCR/TRIDIS | — | En cours |

## Corpus

**CATMuS Medieval** (Clérice et al., 2024) — corpus multilingue de manuscrits médiévaux.
- HuggingFace : `CATMuS/medieval`
- Couverture : XIe–XVIe siècles, vieux français et latin
- Licence : CC BY 4.0

## Justification architecturale

Les CNN présentent trois biais inductifs inadaptés aux manuscrits médiévaux :
- **Invariance à la translation** : la position est porteuse d'information (lettrines, s long)
- **Localité des filtres** : les abréviations nécessitent un contexte global
- **Partage des paramètres** : les statistiques spatiales sont non stationnaires

Les Transformers (TrOCR) résolvent ces problèmes via l'attention globale et le modèle de langue GPT-2.

## Article scientifique

Voir [article.md](article.md) pour l'article complet (8 pages, format ArXiv).

## Équipe

Projet réalisé dans le cadre du module Computer Vision — Promotion MD5 — HETIC 2026.

## Licence

MIT License — voir [LICENSE](LICENSE) pour les détails.

## Références

- Clérice et al. (2024). CATMuS Medieval. ICDAR 2024.
- Kiessling et al. (2019). Kraken. DH 2019.
- Li et al. (2021). TrOCR. arXiv:2109.10282.
- Kirillov et al. (2023). Segment Anything. ICCV 2023.
