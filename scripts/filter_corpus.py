"""
filter_corpus.py
Filtre les images de mauvaise qualité et nettoie les transcriptions.
Objectif : selectionner les 80 meilleures images pour le fine-tuning.
"""

import os
import glob
import shutil
import random
from PIL import Image
import numpy as np

# ── Paramètres de filtrage ───────────────────────────────────────────────────
HAUTEUR_MIN = 30       # pixels
HAUTEUR_MAX = 250      # pixels
LARGEUR_MIN = 200      # pixels
DENSITE_MIN = 0.05     # 5% pixels sombres minimum
DENSITE_MAX = 0.45     # 45% pixels sombres maximum
TEXTE_MIN   = 5        # caractères minimum
TEXTE_MAX   = 500      # pas de limite sur le texte
MAX_IMAGES  = 80       # garder seulement 80 images

# ── Dossiers ─────────────────────────────────────────────────────────────────
DOSSIER_TRAIN   = "data/train"
DOSSIER_FILTRE  = "data/train_filtre"
DOSSIER_REJETE  = "data/train_rejete"

os.makedirs(DOSSIER_FILTRE, exist_ok=True)
os.makedirs(DOSSIER_REJETE, exist_ok=True)


def analyser_image(chemin_img):
    """Analyse une image et retourne ses métriques qualité."""
    try:
        img = Image.open(chemin_img).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        densite = float(np.mean(arr < 128))
        return {"ok": True, "h": h, "w": w, "densite": densite}
    except Exception as e:
        return {"ok": False, "erreur": str(e)}


def analyser_texte(chemin_txt):
    """Analyse une transcription et retourne ses métriques qualité."""
    try:
        with open(chemin_txt, encoding="utf-8") as f:
            texte = f.read().strip()
        n_chars = len(texte)
        n_espaces = texte.count("  ")
        return {"ok": True, "texte": texte, "n_chars": n_chars, "n_espaces": n_espaces}
    except Exception as e:
        return {"ok": False, "erreur": str(e)}


def filtrer_corpus():
    """Filtre le corpus et copie les meilleures images dans data/train_filtre."""

    # Vider les dossiers de sortie
    for f in glob.glob(os.path.join(DOSSIER_FILTRE, "*")):
        os.remove(f)
    for f in glob.glob(os.path.join(DOSSIER_REJETE, "*")):
        os.remove(f)

    images = sorted(glob.glob(os.path.join(DOSSIER_TRAIN, "*.jpg")))
    random.seed(42)
    random.shuffle(images)

    print(f"\n=== Filtrage corpus ===")
    print(f"Images trouvées : {len(images)}")

    stats = {
        "total": len(images),
        "acceptees": 0,
        "rejetees_image": 0,
        "rejetees_texte": 0,
        "rejetees_paire": 0,
    }

    raisons_rejet = []

    for chemin_img in images:

        # Stop si on a assez d'images
        if stats["acceptees"] >= MAX_IMAGES:
            break

        nom_base = os.path.splitext(os.path.basename(chemin_img))[0]
        chemin_txt = os.path.join(DOSSIER_TRAIN, nom_base + ".txt")

        # ── Vérifier que la paire image/texte existe ─────────────────────────
        if not os.path.exists(chemin_txt):
            stats["rejetees_paire"] += 1
            raisons_rejet.append((nom_base, "pas de fichier texte"))
            continue

        # ── Analyser l'image ─────────────────────────────────────────────────
        img_stats = analyser_image(chemin_img)
        if not img_stats["ok"]:
            stats["rejetees_image"] += 1
            raisons_rejet.append((nom_base, f"erreur image: {img_stats['erreur']}"))
            continue

        h, w, densite = img_stats["h"], img_stats["w"], img_stats["densite"]

        raison_img = None
        if h < HAUTEUR_MIN:
            raison_img = f"hauteur trop petite ({h}px < {HAUTEUR_MIN}px)"
        elif h > HAUTEUR_MAX:
            raison_img = f"hauteur trop grande ({h}px > {HAUTEUR_MAX}px)"
        elif w < LARGEUR_MIN:
            raison_img = f"largeur trop petite ({w}px < {LARGEUR_MIN}px)"
        elif densite < DENSITE_MIN:
            raison_img = f"image trop vide ({densite:.1%})"
        elif densite > DENSITE_MAX:
            raison_img = f"image trop dense ({densite:.1%})"

        if raison_img:
            stats["rejetees_image"] += 1
            raisons_rejet.append((nom_base, raison_img))
            continue

        # ── Analyser le texte ────────────────────────────────────────────────
        txt_stats = analyser_texte(chemin_txt)
        if not txt_stats["ok"]:
            stats["rejetees_texte"] += 1
            raisons_rejet.append((nom_base, f"erreur texte"))
            continue

        n_chars = txt_stats["n_chars"]

        if n_chars < TEXTE_MIN:
            stats["rejetees_texte"] += 1
            raisons_rejet.append((nom_base, f"texte trop court ({n_chars} chars)"))
            continue

        # ── Image acceptée ───────────────────────────────────────────────────
        shutil.copy(chemin_img, os.path.join(DOSSIER_FILTRE, os.path.basename(chemin_img)))
        shutil.copy(chemin_txt, os.path.join(DOSSIER_FILTRE, os.path.basename(chemin_txt)))
        stats["acceptees"] += 1

    # ── Rapport ──────────────────────────────────────────────────────────────
    print(f"\n=== Résultats ===")
    print(f"  Total analysé      : {stats['total']}")
    print(f"  Acceptées          : {stats['acceptees']}")
    print(f"  Rejetées (image)   : {stats['rejetees_image']}")
    print(f"  Rejetées (texte)   : {stats['rejetees_texte']}")
    print(f"  Rejetées (paire)   : {stats['rejetees_paire']}")

    if raisons_rejet:
        print(f"\n  Exemples de rejets :")
        for nom, raison in raisons_rejet[:5]:
            print(f"    {nom} → {raison}")

    return stats


if __name__ == "__main__":
    stats = filtrer_corpus()

    images_finales = glob.glob(os.path.join(DOSSIER_FILTRE, "*.jpg"))
    print(f"\n  Images prêtes pour fine-tuning : {len(images_finales)}")

    if len(images_finales) >= MAX_IMAGES:
        print(f"  ✓ {MAX_IMAGES} images sélectionnées — fine-tuning rapide (15-20 min GPU)")
    else:
        print(f"  ⚠ Seulement {len(images_finales)} images acceptées")
