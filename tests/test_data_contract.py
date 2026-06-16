import pytest
import jsonschema
import tempfile
from src.utils.data_contract import valider_data_contract, marquer_needs_review, sha256_fichier


@pytest.fixture
def contrat_valide():
    return {
        "metadata": {
            "corpus": "CATMuS-Medieval",
            "modele": "Gallicorpora+_best.mlmodel",
            "date_production": "2026-06-01",
            "sha256_train": "abc123def456",
            "cer_global": 0.08,
            "wer_global": 0.14,
        },
        "pages": [{
            "page_id": "page_001",
            "image_source": "data/test/raw/page_001.jpg",
            "lignes": [{
                "id": "l0001",
                "transcription": "En cel tens que li rois",
                "confiance": 0.92,
                "needs_review": False,
                "polygone": [[10, 20], [300, 20], [300, 45], [10, 45]],
            }],
        }],
    }


class TestValidationSchema:
    def test_contrat_valide_passe(self, contrat_valide):
        assert valider_data_contract(contrat_valide) is True

    def test_metadata_manquante_echoue(self, contrat_valide):
        del contrat_valide["metadata"]
        with pytest.raises(jsonschema.ValidationError):
            valider_data_contract(contrat_valide)

    def test_confiance_hors_plage_echoue(self, contrat_valide):
        contrat_valide["pages"][0]["lignes"][0]["confiance"] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            valider_data_contract(contrat_valide)


class TestNeedsReview:
    def test_confiance_basse_marquee(self):
        ligne = {"id": "l0001", "transcription": "test", "confiance": 0.5}
        resultat = marquer_needs_review(ligne, seuil=0.7)
        assert resultat["needs_review"] is True

    def test_confiance_haute_non_marquee(self):
        ligne = {"id": "l0001", "transcription": "test", "confiance": 0.9}
        resultat = marquer_needs_review(ligne, seuil=0.7)
        assert resultat["needs_review"] is False


class TestSha256:
    def test_sha256_reproductible(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("contenu de test")
            chemin = f.name
        assert sha256_fichier(chemin) == sha256_fichier(chemin)

    def test_sha256_longueur(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("test")
            chemin = f.name
        assert len(sha256_fichier(chemin)) == 64
