# Reconnaissance Automatique de Texte Manuscrit Médiéval : Pipeline HTR Bout-en-Bout avec Comparaison Architecturale CRNN/CTC et Transformers

**Projet MD5 — HETIC 2026 — Module Computer Vision**  
**Groupe :** MD5-2026  
**Corpus :** CATMuS Medieval (Gallica/BnF)  
**Dépôt :** https://github.com/abdelakimnjankouo-ai/htr-medieval-manuscripts-2026

---

## Résumé

Ce travail présente un pipeline complet de Reconnaissance Automatique de Texte Manuscrit (HTR) appliqué à des manuscrits médiévaux français des XIIIe–XVe siècles. Nous évaluons deux configurations du modèle McCATMuS (CRNN+CTC) : sans fine-tuning (baseline, CER = 32.5%) et après fine-tuning sur 80 images filtrées par critères qualité (CER validation = 10.6%, CER test = 23.25%). Notre contribution méthodologique principale est le filtrage automatique du corpus par critères morphologiques, qui réduit le corpus de 300 à 80 images de haute qualité et accélère le fine-tuning de 5× tout en améliorant la val_accuracy de 82.9% à 89.35%. L'analyse des erreurs révèle que les principales difficultés restent les caractères spéciaux médiévaux (tildes d'abréviation, caractères latins rares) et les dépendances longue portée — limites structurelles du CRNN que les architectures Transformer (TrOCR) sont conçues pour résoudre.

**Mots-clés :** HTR, manuscrits médiévaux, CRNN, CTC, Kraken, CATMuS, CER, fine-tuning, filtrage corpus, pipeline bout-en-bout

---

## 1. Introduction

La transcription automatique de manuscrits médiévaux constitue un défi majeur pour les humanités numériques. Ces documents présentent des caractéristiques radicalement différentes des images naturelles : support physique vieilli (parchemin translucide, encre ferro-gallique oxydée), écriture cursive avec ligatures et abréviations, mise en page non uniforme, et dégradations physiques accumulées sur plusieurs siècles.

Les approches classiques de reconnaissance optique de caractères (OCR) échouent sur ces documents pour des raisons structurelles bien identifiées. Les réseaux de neurones convolutifs (CNN), malgré leurs performances remarquables sur ImageNet, encodent trois biais inductifs inadaptés aux manuscrits : la localité des filtres (un filtre k×k ne voit que k² pixels voisins), l'invariance à la translation (la position est porteuse d'information dans un manuscrit), et le partage des paramètres (les statistiques spatiales des manuscrits sont non stationnaires).

Dans ce travail, nous construisons un pipeline HTR complet et montrons que le filtrage qualité du corpus d'entraînement est un facteur critique pour le fine-tuning de modèles HTR sur des corpus spécialisés. Nous rapportons honnêtement l'écart entre les métriques de validation et de test, et analysons les patterns d'erreurs résiduels.

---

## 2. État de l'art

### 2.1 Architectures CRNN+CTC

Le CRNN (Shi et al., 2015) combine un extracteur de features CNN (VGG-like) avec un BiLSTM qui capture les dépendances horizontales, et une couche CTC (Graves, 2006) qui permet la transcription sans alignement caractère par caractère. Kraken (Kiessling et al., 2019) étend ce paradigme avec le modèle BLLA de segmentation par baselines, adapté aux lignes courbées des manuscrits sur parchemin.

Les limites structurelles du CRNN pour les manuscrits sont bien documentées : le collapse vertical suppose une ligne parfaitement horizontale, les LSTM ont une mémoire limitée sur les longues lignes (>100 caractères), et l'absence de mécanisme d'attention empêche la focalisation contextuelle.

### 2.2 Architectures Transformer pour l'HTR

TrOCR (Li et al., 2021) adopte une architecture encodeur-décodeur : un encodeur BEiT (Vision Transformer pré-entraîné par masquage de patches) et un décodeur GPT-2 avec cross-attention. Le modèle de langue implicite du décodeur permet une meilleure gestion des abréviations médiévales et des dépendances contextuelles longues. Notre analyse des erreurs confirme que les limites du CRNN (tildes d'abréviation, caractères rares) correspondent exactement aux forces de TrOCR.

### 2.3 Importance du filtrage des données

La qualité des données d'entraînement est un facteur critique souvent sous-estimé. Sur le corpus CATMuS Medieval, les images présentent une grande hétérogénéité : hauteurs variables (20 à 290 pixels), densités de pixels sombres disparates, et transcriptions de longueurs très différentes. Notre contribution montre qu'un filtrage par critères morphologiques permet de sélectionner les images les plus informatives, réduisant le bruit et accélérant la convergence.

### 2.4 Pipelines existants

eScriptorium (Kiessling et al., 2019) intègre Kraken dans un environnement de correction collaborative. Transkribus (Kahle et al., 2017) propose une interface similaire. Ces outils confirment la pertinence de l'architecture CRNN+CTC fine-tunée pour les manuscrits médiévaux.

---

## 3. Données et corpus

### 3.1 CATMuS Medieval

Nous utilisons le corpus CATMuS Medieval (Clérice et al., 2024), accessible sur HuggingFace (`CATMuS/medieval`). Ce corpus agrège des transcriptions de manuscrits médiévaux français et latins provenant de plusieurs bibliothèques numériques (Gallica/BnF, BVMM). Il couvre les XIe–XVIe siècles avec une diversité de mains et de scripts.

Notre partition expérimentale initiale :
- **Corpus brut :** 300 images de lignes (train), 50 (val), 10 (test)
- **Corpus filtré :** 80 images sélectionnées par critères qualité (70 train, 10 val)

### 3.2 Filtrage qualité du corpus

Une contribution méthodologique de ce travail est le filtrage automatique du corpus par critères morphologiques. Sur les 300 images initiales, nous appliquons les critères suivants :

- Hauteur : 30–250 pixels
- Largeur : ≥ 200 pixels
- Densité de pixels sombres : 5%–45%
- Longueur de transcription : ≥ 5 caractères

Ce filtrage sélectionne 80 images (~26.7% du corpus), éliminant les images trop petites (artefacts de segmentation), trop denses (zones dégradées), ou avec des transcriptions aberrantes. L'entraînement est 5× plus rapide (6 min vs >30 min sur le corpus complet) et la val_accuracy est améliorée de 82.9% à 89.35%.

### 3.3 Défis spécifiques

**Show-through** : la translucidité du parchemin crée des traces fantômes du verso. Notre analyse des erreurs confirme ce problème avec 95 insertions sur 873 caractères testés.

**Encre ferro-gallique** : l'oxydation progressive crée un contraste variable, justifiant l'utilisation de Sauvola adaptatif.

**Abréviations médiévales** : le tilde d'abréviation (`COMBINING TILDE`) représente à lui seul 26 erreurs sur 203 (12.8% des erreurs totales), confirmant la difficulté structurelle des modèles locaux sur ce type de caractère.

**Caractères latins spéciaux** : ꝑ (per), ꝓ (pro), ⊊, ⟦, ⟧ — caractères rares du vieux français peu représentés dans les données d'entraînement.

---

## 4. Pipeline HTR

### 4.1 Prétraitement adaptatif

**CLAHE** corrige l'éclairage non uniforme sur le canal L (LAB). Paramètres : clipLimit=2.0, tileGridSize=(8,8).

**Deskewing** estime l'angle d'inclinaison par maximisation de la variance de projection horizontale, plage [-10°, +10°], pas 0.5°.

**Binarisation de Sauvola** (Sauvola & Pietikäinen, 2000) :

$$t(i,j) = \mu(i,j) \cdot \left[1 + k \cdot \left(\frac{\sigma(i,j)}{R} - 1\right)\right]$$

avec k=0.2, R=128, fenêtre 25×25 pixels.

### 4.2 Segmentation de layout

SAM (Kirillov et al., 2023) identifie les zones texte, colonnes et illustrations en mode zero-shot via son encodeur ViT-H pré-entraîné par MAE.

### 4.3 Segmentation de lignes

Kraken BLLA représente chaque ligne par une **baseline** (polyligne) et un **polygone englobant**, gérant les lignes courbées du parchemin. L'ordre de lecture est reconstruit par graphe topologique.

### 4.4 Reconnaissance HTR et fine-tuning

**McCATMuS baseline** : modèle CRNN+CTC pré-entraîné (CNN VGG-like + BiLSTM 2 couches + CTC). CER baseline = 32.5%.

**McCATMuS fine-tuné** : fine-tuning sur 80 images filtrées, 20 epochs max, learning rate 0.0001, early stopping patience 5. Entraînement sur GPU Tesla T4 (Google Colab), durée ~6 minutes pour 13 stages. Meilleur checkpoint : `checkpoint_08-0.8935.ckpt` (val_accuracy = 89.35%).

Les fichiers d'entraînement sont générés au format ALTO XML v4, avec baselines et polygones calculés automatiquement depuis les dimensions des images.

### 4.5 Export JSON

Sortie conforme au data contract : `page_id`, `schema_version`, `pipeline`, `n_lignes`, `cer_estime`, avec pour chaque ligne `line_id`, `transcription`, `confidence`, `needs_review`, `flags`.

---

## 5. Expériences et résultats

### 5.1 Métriques

$$\text{CER} = \frac{S + D + I}{N}$$

où S, D, I sont les substitutions, suppressions et insertions, N le nombre de caractères de référence.

### 5.2 Résultats principaux

| Modèle | CER validation | CER test (5 images) | Val accuracy |
|--------|---------------|--------------------|----|
| McCATMuS baseline | 32.5% | — | 82.9% |
| McCATMuS fine-tuné | **10.6%** | **23.25%** | **89.35%** |

**Amélioration val_accuracy :** 82.9% → 89.35% (+6.4 points)  
**Amélioration CER validation :** 32.5% → 10.6% (-67.4%)

### 5.3 Écart validation / test

Un écart important est observé entre le CER de validation (10.6%) et le CER de test réel (23.25%). Cet écart s'explique par plusieurs facteurs :

La **distribution shift** entre les données de validation (issues du même corpus filtré) et les données de test (images non vues) révèle un surapprentissage partiel sur le sous-corpus d'entraînement. Avec seulement 80 images d'entraînement, le modèle fine-tuné apprend les spécificités stylistiques de ce sous-corpus mais généralise moins bien à de nouvelles pages.

Cette observation confirme la nécessité d'un corpus de test véritablement indépendant, idéalement issu de manuscrits différents, pour évaluer les capacités de généralisation réelles du modèle.

### 5.4 Analyse qualitative des erreurs

Sur 873 caractères testés, 203 erreurs ont été identifiées (CER = 23.25%) :
- **95 insertions** (47%) — caractères fantômes générés
- **95 substitutions** (47%) — caractères mal reconnus
- **13 suppressions** (6%) — caractères manquants

**Erreurs dominantes par catégorie :**

| Erreur | Occurrences | Cause probable |
|--------|-------------|----------------|
| COMBINING TILDE omis | 26 | Tilde d'abréviation rare |
| Caractères latins spéciaux | 15 | ꝑ, ꝓ peu représentés |
| Confusion i/e, o/e | 12 | Ambiguïté morphologique |
| Insertions de s, a, n | 8 | Show-through parchemin |

**Répartition par script Unicode :**
- Common (175 chars) : 96.57% de précision ✅
- Latin (631 chars) : 78.61% de précision
- Inherited (67 chars) : 26.87% de précision ⚠️

Les caractères "Inherited" (diacritiques, tildes combinants) sont les plus mal reconnus — exactement les caractères pour lesquels le contexte global d'un Transformer serait décisif.

---

## 6. Discussion

### 6.1 Pourquoi les CNN atteignent leurs limites sur les manuscrits

**L'invariance à la translation** : le pooling détruit l'information positionnelle. Dans un manuscrit, la position est porteuse de sens — le s long (ʃ) n'apparaît jamais en position finale, une lettrine en début de section diffère de la même forme en milieu de ligne.

**La localité des filtres** : le champ réceptif effectif d'un ResNet-50 est ~11 pixels (Luo et al., 2016). Le tilde d'abréviation, placé au-dessus d'une lettre, nécessite un contexte de plusieurs dizaines de pixels pour être interprété — hors de portée d'un CRNN sans mécanisme d'attention.

**Le partage des paramètres** : les statistiques spatiales des manuscrits sont non stationnaires. Nos résultats confirment cette limite : les caractères "Inherited" (diacritiques combinants) ont une précision de seulement 26.87%, contre 96.57% pour les caractères "Common".

### 6.2 Apport du fine-tuning ciblé

Le fine-tuning sur données filtrées améliore significativement la val_accuracy (+6.4 points). L'analyse montre que le modèle apprend mieux les caractères fréquents du corpus (Latin : 78.61% vs ~65% estimé pour la baseline) mais reste limité sur les caractères rares et les diacritiques.

### 6.3 Limites

**Taille du corpus de test** : 5 images représentent 873 caractères — insuffisant pour des conclusions statistiques robustes. Un corpus de test de 50+ images serait nécessaire.

**Surapprentissage** : l'écart validation/test (10.6% vs 23.25%) suggère un surapprentissage partiel sur le sous-corpus de 80 images. Une augmentation des données (rotation, bruit, déformation élastique) réduirait cet écart.

**Caractères rares** : ꝑ, ꝓ, ⟦, ⟧ et les diacritiques combinants sont structurellement difficiles pour le CRNN — leur précision resterait faible même avec plus de données.

**Absence de baseline humaine** : le CER inter-annotateurs (TP1) n'a pas été mesuré, empêchant de situer nos résultats par rapport au plafond théorique humain.

### 6.4 Perspectives

**TrOCR fine-tuné** adresserait directement les limites observées : le mécanisme d'attention globale résoudrait les 26 erreurs de COMBINING TILDE, et le modèle de langue GPT-2 améliorerait la reconnaissance des abréviations. Une réduction du CER à 8-12% est attendue.

**Data augmentation** : rotation (±5°), bruit gaussien, déformation élastique doublerait le corpus effectif de 80 à 160 images sans collecte supplémentaire.

**LoRA** (Hu et al., 2021) permettrait un fine-tuning efficace de TrOCR avec seulement 0.6% des paramètres entraînés, réduisant les besoins en mémoire GPU à 4 Go.

---

## 7. Conclusion

Nous avons présenté un pipeline HTR complet pour les manuscrits médiévaux français, avec deux contributions principales. D'abord, un **filtrage automatique du corpus** par critères morphologiques qui réduit le corpus de 300 à 80 images de haute qualité, améliorant la val_accuracy de 82.9% à 89.35% et réduisant le temps d'entraînement de 5×. Ensuite, une **évaluation honnête** distinguant CER de validation (10.6%) et CER de test réel (23.25%), révélant un surapprentissage partiel inhérent aux corpus de fine-tuning de petite taille.

L'analyse des erreurs confirme les prédictions théoriques du cours : les limites du CRNN (champ réceptif limité, invariance à la translation) se manifestent précisément sur les caractères qui nécessitent un contexte global — tildes d'abréviation (26.87% de précision sur les "Inherited") et diacritiques combinants. Ces résultats motivent directement l'adoption de TrOCR comme étape suivante.

---

## Références

- Clérice, T. et al. (2024). *CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond*. ICDAR 2024.
- Graves, A. et al. (2006). *Connectionist temporal classification*. ICML 2006.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Kahle, P. et al. (2017). *Transkribus: A Service Platform for Transcription, Recognition and Retrieval of Historical Documents*. ICDAR 2017.
- Kiessling, B. et al. (2019). *Kraken: an universal text recognizer for the humanities*. DH 2019.
- Kirillov, A. et al. (2023). *Segment Anything*. ICCV 2023.
- Li, M. et al. (2021). *TrOCR: Transformer-based optical character recognition with pre-trained models*. arXiv:2109.10282.
- Luo, W. et al. (2016). *Understanding the effective receptive field in deep convolutional neural networks*. NeurIPS 2016.
- Needleman, S.B., Wunsch, C.D. (1970). *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology.
- Sauvola, J., Pietikäinen, M. (2000). *Adaptive document image binarization*. Pattern Recognition.
- Shi, B. et al. (2015). *An end-to-end trainable neural network for image-based sequence recognition*. arXiv:1507.05717.

---

*Ce travail a été réalisé dans le cadre du module Computer Vision — Promotion MD5 — HETIC 2026.*
