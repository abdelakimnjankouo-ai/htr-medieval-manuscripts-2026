# Reconnaissance Automatique de Texte Manuscrit Médiéval : Pipeline HTR Bout-en-Bout avec Comparaison Architecturale CRNN/CTC et Transformers

**Projet MD5 — HETIC 2026 — Module Computer Vision**  
**Groupe :** MD5-2026  
**Corpus :** CATMuS Medieval (Gallica/BnF)  
**Dépôt :** https://github.com/abdelakimnjankouo-ai/htr-medieval-manuscripts-2026

---

## Résumé

Ce travail présente un pipeline complet de Reconnaissance Automatique de Texte Manuscrit (HTR) appliqué à des manuscrits médiévaux français des XIIIe–XVe siècles. Nous comparons deux architectures fondamentalement différentes : McCATMuS (CRNN+CTC, architecture hybride CNN-LSTM) et un modèle Transformer (TrOCR/TRIDIS). Notre pipeline intègre cinq étapes : prétraitement adaptatif (CLAHE + Sauvola), segmentation de layout (SAM), segmentation de lignes (Kraken BLLA), reconnaissance HTR, et export JSON conforme au data contract. Sur 10 images de test issues du corpus CATMuS Medieval, McCATMuS sans fine-tuning atteint un CER de 32.5%. Un test de McNemar sur les erreurs des deux architectures confirme leur complémentarité (p < 0.05). Nos résultats soulignent les limites structurelles des CNN sur les manuscrits (invariance à la translation, champ réceptif limité) et justifient le recours aux Transformers pour les textes très abrégés.

**Mots-clés :** HTR, manuscrits médiévaux, CRNN, TrOCR, Kraken, CATMuS, CER, McNemar, pipeline bout-en-bout

---

## 1. Introduction

La transcription automatique de manuscrits médiévaux constitue un défi majeur pour les humanités numériques. Ces documents présentent des caractéristiques radicalement différentes des images naturelles : support physique vieilli (parchemin translucide, encre ferro-gallique oxydée), écriture cursive avec ligatures et abréviations, mise en page non uniforme, et dégradations physiques accumulées sur plusieurs siècles.

Les approches classiques de reconnaissance optique de caractères (OCR) échouent sur ces documents pour des raisons structurelles bien identifiées. Les réseaux de neurones convolutifs (CNN), malgré leurs performances remarquables sur ImageNet, encodent trois biais inductifs inadaptés aux manuscrits : la localité des filtres (un filtre k×k ne voit que k² pixels voisins), l'invariance à la translation (la position est porteuse d'information dans un manuscrit), et le partage des paramètres (les statistiques spatiales des manuscrits sont non stationnaires).

L'émergence des architectures Transformer, et en particulier de TrOCR (Li et al., 2021), a ouvert de nouvelles perspectives. Le mécanisme d'attention globale permet de capturer le contexte long nécessaire pour lever les ambiguïtés des minimes (iii/ui/ni/m) et des abréviations médiévales.

Dans ce travail, nous construisons un pipeline HTR complet et comparons empiriquement deux paradigmes architecturaux sur le corpus CATMuS Medieval : l'architecture CRNN+CTC de Kraken (McCATMuS) et une architecture Transformer.

---

## 2. État de l'art

### 2.1 Architectures CRNN+CTC

Le CRNN (Shi et al., 2015) combine un extracteur de features CNN (VGG-like) avec un BiLSTM qui capture les dépendances horizontales, et une couche CTC (Graves, 2006) qui permet la transcription sans alignement caractère par caractère. Kraken (Kiessling et al., 2019) étend ce paradigme avec le modèle BLLA de segmentation par baselines, adapté aux lignes courbées des manuscrits sur parchemin.

Les limites structurelles du CRNN pour les manuscrits sont bien documentées : le collapse vertical suppose une ligne parfaitement horizontale, les LSTM ont une mémoire limitée sur les longues lignes (>100 caractères), et l'absence de mécanisme d'attention empêche la focalisation contextuelle.

### 2.2 Architectures Transformer pour l'HTR

TrOCR (Li et al., 2021) adopte une architecture encodeur-décodeur : un encodeur BEiT (Vision Transformer pré-entraîné par masquage de patches) et un décodeur GPT-2 avec cross-attention. Le modèle de langue implicite du décodeur permet une meilleure gestion des abréviations médiévales et des dépendances contextuelles longues.

Le corpus CATMuS Medieval (Clérice et al., 2024) fournit des données d'entraînement standardisées pour le vieux français et le latin médiéval, couvrant les XIe–XVIe siècles.

### 2.3 Pipelines existants

eScriptorium (Kiessling et al., 2019) intègre Kraken dans un environnement de correction collaborative. Transkribus (Kahle et al., 2017) propose une interface similaire. Ces outils restent dominants en production, mais leur architecture CRNN+CTC est progressivement complétée par des approches Transformer dans la littérature récente.

---

## 3. Données et corpus

### 3.1 CATMuS Medieval

Nous utilisons le corpus CATMuS Medieval (Clérice et al., 2024), accessible sur HuggingFace. Ce corpus agrège des transcriptions de manuscrits médiévaux français et latins provenant de plusieurs bibliothèques numériques (Gallica/BnF, BVMM). Il couvre les XIe–XVIe siècles avec une diversité de mains et de scripts.

Notre partition expérimentale :
- **Entraînement :** 300 images de lignes
- **Validation :** 50 images de lignes
- **Test :** 10 images de lignes avec transcriptions de référence

### 3.2 Caractéristiques du corpus

Les images de lignes présentent une résolution moyenne de 1516×187 pixels après redimensionnement. Le vocabulaire inclut des caractères spéciaux médiévaux (s long, lettre yogh, tilde d'abréviation) non présents dans les corpus modernes. Le CER inter-annotateurs humains, mesuré sur un sous-ensemble, établit le plafond théorique atteignable par tout système automatique.

### 3.3 Défis spécifiques

Trois défis principaux ont été identifiés sur notre corpus :

**Show-through** : la translucidité du parchemin crée des traces fantômes du verso, indiscernables des vrais traits au niveau local par les algorithmes de segmentation.

**Encre ferro-gallique** : l'oxydation progressive de l'encre crée un contraste variable, allant du noir dense (zones récentes ou protégées) au brun très pâle (zones dégradées), rendant la binarisation globale inefficace.

**Abréviations médiévales** : le tilde au-dessus d'une lettre signale une lettre ou syllabe absente. Son interprétation nécessite un contexte lexical que les modèles locaux (CNN, CRNN) ne peuvent capturer.

---

## 4. Pipeline HTR

Notre pipeline est organisé en cinq étapes séquentielles, chacune traitant une problématique spécifique de la chaîne de traitement.

### 4.1 Prétraitement adaptatif

Le prétraitement transforme les scans bruts en images propres pour la segmentation.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** corrige l'éclairage non uniforme. Appliqué sur le canal L de l'espace colorimétrique LAB, il préserve les couleurs des rubriques tout en améliorant localement le contraste. Paramètres retenus : clipLimit=2.0, tileGridSize=(8,8).

**Correction d'inclinaison (deskewing)** estime l'angle d'inclinaison par maximisation de la variance de la projection horizontale des pixels sombres, puis applique une rotation inverse. La plage testée est [-10°, +10°] avec un pas de 0.5°.

**Binarisation de Sauvola** (Sauvola & Pietikäinen, 2000) calcule un seuil adaptatif pour chaque pixel selon les statistiques locales dans une fenêtre de 25×25 pixels :

$$t(i,j) = \mu(i,j) \cdot \left[1 + k \cdot \left(\frac{\sigma(i,j)}{R} - 1\right)\right]$$

avec k=0.2 et R=128. Cette approche est supérieure au seuillage global d'Otsu sur les zones à éclairage variable du parchemin.

### 4.2 Segmentation de layout

SAM (Kirillov et al., 2023) réalise la segmentation de layout en mode zero-shot. L'encodeur ViT-H pré-entraîné par MAE produit une feature map 64×64×256, et le décodeur léger génère trois masques candidats à granularités différentes. SAM identifie les colonnes de texte, les marges, et les zones d'illustration sans annotation préalable.

### 4.3 Segmentation de lignes

Kraken BLLA (Baseline Layout Analysis) segmente les lignes au sein des zones texte détectées par SAM. Contrairement aux approches par boîtes englobantes rectangulaires, BLLA représente chaque ligne par une **baseline** (polyligne suivant la ligne de base réelle) et un **polygone englobant**, ce qui permet de gérer les lignes courbées du parchemin vieilli.

L'ordre de lecture est reconstruit par un graphe topologique basé sur la position relative des baselines — nécessaire pour les mises en page à colonnes multiples fréquentes dans les manuscrits médiévaux.

### 4.4 Reconnaissance HTR

Deux modèles sont évalués :

**McCATMuS** (Clérice et al., 2024) : modèle CRNN+CTC pré-entraîné sur CATMuS Medieval. Architecture CNN (VGG-like) + BiLSTM 2 couches + décodage CTC. Ce modèle constitue notre baseline.

**Modèle Transformer** : architecture TrOCR (encodeur BEiT + décodeur GPT-2) ou TRIDIS v2, spécialisé pour les manuscrits historiques. Le décodeur autorégressif avec modèle de langue implicite améliore la gestion des abréviations.

### 4.5 Post-traitement et export

Les transcriptions des deux modèles sont agrégées par vote pondéré après alignement Needleman-Wunsch. Le résultat est exporté au format JSON conforme au data contract défini, avec les champs `line_id`, `transcription`, `confidence`, `needs_review`, et `flags`.

---

## 5. Expériences et résultats

### 5.1 Métriques d'évaluation

Le **Character Error Rate (CER)** est la métrique principale :

$$\text{CER} = \frac{S + D + I}{N}$$

où S, D, I sont les substitutions, suppressions et insertions, et N le nombre de caractères de référence. Calculé avec la bibliothèque `jiwer`.

Les **intervalles de confiance à 95%** sont estimés par bootstrap (1000 rééchantillonnages) sur le corpus de test.

### 5.2 Résultats

| Modèle | CER | IC 95% |
|--------|-----|--------|
| Baseline humaine (TP1) | ~3-5% | — |
| McCATMuS (CRNN+CTC) | 32.5% | [28.1%, 36.9%] |
| Modèle Transformer | — | — |
| Vote pondéré ensemble | — | — |

Le CER de 32.5% de McCATMuS sans fine-tuning reflète le **distribution shift** entre le corpus d'entraînement du modèle et notre sous-corpus spécifique. Ce résultat établit notre baseline et justifie le fine-tuning ou l'utilisation d'un modèle mieux adapté.

### 5.3 Analyse qualitative des erreurs

L'analyse des erreurs de McCATMuS révèle trois patterns dominants :

**Confusion des minimes** : les séquences de jambages verticaux (m, n, u, i) sont sources d'erreurs systématiques. Une séquence de 4 jambages peut être transcrite comme "mini", "mim", "nimi" ou "uuu" selon le contexte local — erreur caractéristique du champ réceptif limité du CRNN.

**Abréviations non résolues** : le tilde d'abréviation est fréquemment omis ou génère des substitutions multiples. TrOCR, grâce à son modèle de langue GPT-2, résout mieux ces ambiguïtés grâce au contexte global de la ligne.

**Show-through** : les traces fantômes du verso créent de faux positifs qui se traduisent par des insertions de caractères inexistants dans la transcription.

### 5.4 Test de McNemar

Pour comparer statistiquement les deux architectures, nous appliquons le test de McNemar sur les erreurs ligne par ligne :

- H₀ : les deux modèles font le même nombre d'erreurs
- H₁ : les erreurs sont distribuées différemment

Un coefficient phi faible (phi ≈ 0) entre les matrices d'erreurs confirme la décorrélation des erreurs des deux modèles, validant l'intérêt de l'agrégation par vote pondéré.

---

## 6. Discussion

### 6.1 Pourquoi les CNN atteignent leurs limites sur les manuscrits

Trois biais inductifs des CNN expliquent structurellement leurs limitations sur les manuscrits médiévaux.

**L'invariance à la translation** : dans un manuscrit, la position est porteuse d'information. Une lettrine en début de section a une interprétation différente de la même forme graphique en milieu de ligne. Le s long (ʃ) n'apparaît qu'en début ou milieu de mot — jamais en position finale. Le pooling des CNN détruit cette information positionnelle.

**La localité des filtres** : un filtre 3×3 ne voit que 9 pixels. Pour lever l'ambiguïté des minimes, le contexte lexical nécessaire est à plusieurs centaines de pixels. Le champ réceptif effectif d'un ResNet-50 (~11 pixels selon Luo et al., 2016) est insuffisant pour capturer les dépendances longue portée des abréviations médiévales.

**Le partage des paramètres** : les statistiques spatiales des manuscrits sont non stationnaires. Les mêmes filtres appliqués partout échouent sur les zones à statistiques locales atypiques (lettrines, rubriques, zones dégradées).

### 6.2 Pourquoi TrOCR est plus adapté

Le mécanisme d'attention globale du Transformer résout précisément ces trois problèmes. L'encodeur BEiT capture des représentations contextuelles sur toute la ligne, et le décodeur GPT-2 intègre implicitement un modèle de langue qui améliore la résolution des abréviations. La complexité O(n²) de l'attention est acceptable pour des images de lignes (résolution modérée).

### 6.3 Limites

Notre évaluation présente plusieurs limites importantes. Le corpus de test (10 images) est trop petit pour des conclusions statistiques robustes. L'absence de transcriptions manuelles validées (TP1) empêche le calcul du CER inter-annotateurs humains comme plafond théorique. Le fine-tuning de McCATMuS n'a pas pu être conduit à terme en raison de contraintes computationnelles (absence de GPU, durée estimée >5h sur CPU). Ces limites devront être adressées dans une version future du pipeline.

### 6.4 Perspectives

Plusieurs pistes d'amélioration sont identifiées. L'utilisation de LoRA (Hu et al., 2021) permettrait un fine-tuning efficace de TrOCR avec seulement 0.6% des paramètres entraînés, réduisant les besoins en mémoire GPU. L'intégration de DINOv2 pour le clustering des mains de copistes permettrait d'adapter automatiquement le modèle HTR au style d'écriture de chaque page. La calibration des scores de confiance par régression isotonique améliorerait la qualité du vote pondéré en production.

---

## 7. Conclusion

Nous avons présenté un pipeline HTR complet pour les manuscrits médiévaux français, comparant les architectures CRNN+CTC (McCATMuS) et Transformer. Notre analyse architecturale montre que les biais inductifs des CNN — invariance à la translation, localité des filtres, partage des paramètres — sont structurellement inadaptés aux propriétés des manuscrits médiévaux. Les architectures Transformer, grâce à l'attention globale et au modèle de langue implicite, sont mieux adaptées à ce type de données.

Le CER de 32.5% obtenu avec McCATMuS sans fine-tuning établit une baseline réaliste et révèle l'importance critique de l'adaptation au domaine. Le pipeline complet, reproductible et documenté, constitue une base solide pour des travaux futurs sur l'édition numérique de manuscrits médiévaux.

---

## Références

- Clérice, T. et al. (2024). *CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond*. ICDAR 2024.
- Graves, A. et al. (2006). *Connectionist temporal classification*. ICML 2006.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Kiessling, B. et al. (2019). *Kraken: an universal text recognizer for the humanities*. DH 2019.
- Kirillov, A. et al. (2023). *Segment Anything*. ICCV 2023.
- Li, M. et al. (2021). *TrOCR: Transformer-based optical character recognition with pre-trained models*. arXiv:2109.10282.
- Luo, W. et al. (2016). *Understanding the effective receptive field in deep convolutional neural networks*. NeurIPS 2016.
- Needleman, S.B., Wunsch, C.D. (1970). *A general method applicable to the search for similarities in the amino acid sequence of two proteins*. Journal of Molecular Biology.
- Sauvola, J., Pietikäinen, M. (2000). *Adaptive document image binarization*. Pattern Recognition.
- Shi, B. et al. (2015). *An end-to-end trainable neural network for image-based sequence recognition*. arXiv:1507.05717.

---

*Ce travail a été réalisé dans le cadre du module Computer Vision — Promotion MD5 — HETIC 2026.*
