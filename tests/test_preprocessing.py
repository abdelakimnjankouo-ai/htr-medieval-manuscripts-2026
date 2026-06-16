import pytest
import numpy as np
from PIL import Image
from src.preprocessing.pipeline import pretraiter_image, _appliquer_clahe, _binariser_sauvola, _corriger_inclinaison


@pytest.fixture
def image_grise_test():
    arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def image_rgb_test():
    arr = np.random.randint(100, 250, (400, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestPretraitementComplet:
    def test_sortie_est_pil_image(self, image_rgb_test):
        resultat = pretraiter_image(image_rgb_test)
        assert isinstance(resultat, Image.Image)

    def test_mode_sortie_est_rgb(self, image_rgb_test):
        resultat = pretraiter_image(image_rgb_test)
        assert resultat.mode == "RGB"

    def test_dimensions_coherentes(self, image_rgb_test):
        w_orig, h_orig = image_rgb_test.size
        resultat = pretraiter_image(image_rgb_test, deskew=False)
        w, h = resultat.size
        assert w > 0 and h > 0

    def test_redimensionnement_taille_max(self):
        grande_image = Image.fromarray(np.zeros((5000, 4000, 3), dtype=np.uint8), mode="RGB")
        resultat = pretraiter_image(grande_image, taille_max=2000, deskew=False)
        assert max(resultat.size) <= 2000


class TestCLAHE:
    def test_forme_inchangee(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _appliquer_clahe(arr)
        assert resultat.shape == arr.shape

    def test_type_uint8(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _appliquer_clahe(arr)
        assert resultat.dtype == np.uint8

    def test_plage_valeurs(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _appliquer_clahe(arr)
        assert resultat.min() >= 0
        assert resultat.max() <= 255


class TestBinarisationSauvola:
    def test_valeurs_binaires(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _binariser_sauvola(arr)
        valeurs_uniques = set(np.unique(resultat))
        assert valeurs_uniques.issubset({0, 255})

    def test_forme_inchangee(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _binariser_sauvola(arr)
        assert resultat.shape == arr.shape

    def test_window_size_pair_corrige(self, image_grise_test):
        arr = np.array(image_grise_test)
        resultat = _binariser_sauvola(arr, window_size=24)
        assert resultat is not None


class TestDeskewing:
    def test_retourne_pil_image(self, image_rgb_test):
        resultat = _corriger_inclinaison(image_rgb_test)
        assert isinstance(resultat, Image.Image)
