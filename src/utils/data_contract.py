import json
import hashlib
from pathlib import Path
import jsonschema

SCHEMA_DATA_CONTRACT = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "HTR Output Data Contract",
    "type": "object",
    "required": ["metadata", "pages"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["corpus", "modele", "date_production", "sha256_train"],
            "properties": {
                "corpus":          {"type": "string"},
                "modele":          {"type": "string"},
                "date_production": {"type": "string"},
                "sha256_train":    {"type": "string"},
                "cer_global":      {"type": "number"},
                "wer_global":      {"type": "number"},
            }
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["page_id", "image_source", "lignes"],
                "properties": {
                    "page_id":      {"type": "string"},
                    "image_source": {"type": "string"},
                    "lignes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "transcription", "confiance", "polygone"],
                            "properties": {
                                "id":            {"type": "string"},
                                "transcription": {"type": "string"},
                                "confiance":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "needs_review":  {"type": "boolean"},
                                "polygone": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


def valider_data_contract(data: dict) -> bool:
    """Valide un dictionnaire contre le schema du data contract.

    Args:
        data: Dictionnaire a valider.

    Returns:
        True si valide.

    Raises:
        jsonschema.ValidationError: Si la validation echoue.
    """
    jsonschema.validate(instance=data, schema=SCHEMA_DATA_CONTRACT)
    return True


def sha256_fichier(chemin) -> str:
    """Calcule le hash SHA-256 d'un fichier.

    Args:
        chemin: Chemin vers le fichier.

    Returns:
        Hash SHA-256 en hexadecimal.
    """
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(65536), b""):
            h.update(bloc)
    return h.hexdigest()


def marquer_needs_review(ligne: dict, seuil: float = 0.7) -> dict:
    """Marque une ligne comme necessitant une revision humaine.

    Args:
        ligne: Dictionnaire d'une ligne HTR avec cle 'confiance'.
        seuil: Seuil de confiance (defaut : 0.7).

    Returns:
        La ligne avec le champ 'needs_review' mis a jour.
    """
    ligne["needs_review"] = ligne.get("confiance", 0.0) < seuil
    return ligne
