import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.filters import threshold_sauvola


def pretraiter_image(chemin_ou_image, deskew=True, clahe=True, binariser=True, debruiter=True, taille_max=4000, angle_max=10.0):
    """Applique la chaine complete de pretraitement a une image de manuscrit.

    Args:
        chemin_ou_image: Chemin vers l'image ou objet PIL.Image.
        deskew: Applique la correction d'inclinaison si True.
        clahe: Applique CLAHE si True.
        binariser: Applique la binarisation Sauvola si True.
        debruiter: Applique le filtre median si True.
        taille_max: Taille maximale en pixels.
        angle_max: Angle maximal pour le deskew.

    Returns:
        Image PIL pretraitee en mode RGB.
    """
    if isinstance(chemin_ou_image, (str, Path)):
        image = Image.open(chemin_ou_image).convert("RGB")
    else:
        image = chemin_ou_image.convert("RGB")

    largeur, hauteur = image.size
    if max(largeur, hauteur) > taille_max:
        ratio = taille_max / max(largeur, hauteur)
        image = image.resize((int(largeur * ratio), int(hauteur * ratio)), Image.LANCZOS)

    if deskew:
        image = _corriger_inclinaison(image, angle_max=angle_max)

    gris = np.array(image.convert("L"))

    if clahe:
        gris = _appliquer_clahe(gris)

    if debruiter:
        gris = cv2.medianBlur(gris, 3)

    if binariser:
        gris = _binariser_sauvola(gris)

    return Image.fromarray(gris).convert("RGB")


def _corriger_inclinaison(image_pil, angle_max=10.0):
    """Corrige l'inclinaison d'une image.

    Args:
        image_pil: Image PIL.
        angle_max: Angle maximum tolere.

    Returns:
        Image PIL redressee.
    """
    gris = np.array(image_pil.convert("L"))
    _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binaire > 0))
    if len(coords) < 100:
        return image_pil

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > angle_max:
        return image_pil

    if abs(angle) > 0.3:
        image_pil = image_pil.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    return image_pil


def _appliquer_clahe(gris, clip_limit=2.0, tile_size=8):
    """Applique CLAHE sur une image en niveaux de gris.

    Args:
        gris: Image en niveaux de gris (uint8).
        clip_limit: Limite de contraste.
        tile_size: Taille de la grille.

    Returns:
        Image normalisee.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gris)


def _binariser_sauvola(gris, window_size=25, k=0.2):
    """Binarisation adaptative Sauvola.

    Args:
        gris: Image en niveaux de gris (uint8).
        window_size: Taille de la fenetre locale.
        k: Parametre de ponderation.

    Returns:
        Image binarisee (uint8).
    """
    if window_size % 2 == 0:
        window_size += 1
    seuil = threshold_sauvola(gris, window_size=window_size, k=k)
    return ((gris > seuil).astype(np.uint8) * 255)
