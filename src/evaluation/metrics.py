import numpy as np
from jiwer import cer, wer


def calculer_cer_wer(hypotheses: list, references: list) -> dict:
    """Calcule le CER et WER globaux.

    Args:
        hypotheses: Transcriptions automatiques.
        references: Transcriptions de reference.

    Returns:
        Dict avec 'cer', 'wer', 'n_lignes'.

    Example:
        >>> calculer_cer_wer(["bonjur"], ["bonjour"])
        {'cer': 0.142, 'wer': 1.0, 'n_lignes': 1}
    """
    n = min(len(hypotheses), len(references))
    if n == 0:
        return {"cer": 1.0, "wer": 1.0, "n_lignes": 0}

    texte_hyp = "\n".join(hypotheses[:n])
    texte_ref = "\n".join(references[:n])

    return {
        "cer": round(cer(texte_ref, texte_hyp), 4),
        "wer": round(wer(texte_ref, texte_hyp), 4),
        "n_lignes": n,
    }


def bootstrap_cer(hypotheses: list, references: list, n_iter: int = 1000, niveau: float = 0.95, seed: int = 42) -> dict:
    """Calcule l'intervalle de confiance bootstrap sur le CER.

    Args:
        hypotheses: Transcriptions automatiques.
        references: Transcriptions de reference.
        n_iter: Nombre d'iterations bootstrap.
        niveau: Niveau de confiance.
        seed: Seed pour la reproductibilite.

    Returns:
        Dict avec 'cer_moyen', 'ic_bas', 'ic_haut', 'niveau'.
    """
    rng = np.random.default_rng(seed)
    n = min(len(hypotheses), len(references))
    cer_global = calculer_cer_wer(hypotheses, references)["cer"]

    cer_echantillons = []
    for _ in range(n_iter):
        indices = rng.integers(0, n, size=n)
        hyp_s = [hypotheses[i] for i in indices]
        ref_s = [references[i] for i in indices]
        cer_echantillons.append(calculer_cer_wer(hyp_s, ref_s)["cer"])

    alpha = 1 - niveau
    return {
        "cer_moyen": round(cer_global, 4),
        "ic_bas":    round(float(np.percentile(cer_echantillons, 100 * alpha / 2)), 4),
        "ic_haut":   round(float(np.percentile(cer_echantillons, 100 * (1 - alpha / 2))), 4),
        "niveau":    niveau,
        "n_iter":    n_iter,
    }


def comparer_mcnemar(erreurs_modele_a: list, erreurs_modele_b: list) -> dict:
    """Test de McNemar pour comparer deux modeles HTR.

    Args:
        erreurs_modele_a: Booléens - True si modele A a fait une erreur.
        erreurs_modele_b: Booléens - True si modele B a fait une erreur.

    Returns:
        Dict avec 'b', 'c', 'statistique', 'p_value', 'significatif'.
    """
    n = min(len(erreurs_modele_a), len(erreurs_modele_b))
    b = sum(1 for i in range(n) if erreurs_modele_a[i] and not erreurs_modele_b[i])
    c = sum(1 for i in range(n) if not erreurs_modele_a[i] and erreurs_modele_b[i])

    if b + c == 0:
        return {"b": 0, "c": 0, "statistique": 0.0, "p_value": 1.0, "significatif": False}

    statistique = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import norm
    p_value = float(1 - norm.cdf(np.sqrt(statistique))) * 2

    return {
        "b": b,
        "c": c,
        "statistique": round(statistique, 4),
        "p_value": round(p_value, 4),
        "significatif": p_value < 0.05,
    }


def taux_needs_review(resultats: list) -> float:
    """Calcule le taux de lignes marquees needs_review.

    Args:
        resultats: Liste de dicts HTR avec cle 'needs_review'.

    Returns:
        Fraction de lignes a reviser.
    """
    if not resultats:
        return 0.0
    return sum(1 for r in resultats if r.get("needs_review", False)) / len(resultats)
