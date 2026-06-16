import pytest
from src.evaluation.metrics import calculer_cer_wer, bootstrap_cer, comparer_mcnemar, taux_needs_review


class TestCerWer:
    def test_transcription_parfaite(self):
        res = calculer_cer_wer(["bonjour monde"], ["bonjour monde"])
        assert res["cer"] == 0.0
        assert res["wer"] == 0.0

    def test_transcription_vide(self):
        res = calculer_cer_wer([], [])
        assert res["cer"] == 1.0
        assert res["n_lignes"] == 0

    def test_erreur_simple(self):
        res = calculer_cer_wer(["bonjur"], ["bonjour"])
        assert res["cer"] > 0.0
        assert res["cer"] < 1.0

    def test_n_lignes_retourne(self):
        res = calculer_cer_wer(["a", "b", "c"], ["a", "b", "c"])
        assert res["n_lignes"] == 3

    def test_seuil_validation(self):
        hyp = ["le roi de france", "en lan de grace"]
        ref = ["le roi de france", "en l'an de grace"]
        res = calculer_cer_wer(hyp, ref)
        assert res["cer"] < 0.15


class TestBootstrapCer:
    def test_structure_retour(self):
        res = bootstrap_cer(["abc"], ["abc"], n_iter=100)
        assert "cer_moyen" in res
        assert "ic_bas" in res
        assert "ic_haut" in res

    def test_ic_coherent(self):
        hyp = ["bonjour"] * 20
        ref = ["bonjour"] * 20
        res = bootstrap_cer(hyp, ref, n_iter=200)
        assert res["ic_bas"] <= res["cer_moyen"] <= res["ic_haut"]

    def test_reproductible(self):
        hyp = ["test errur"] * 10
        ref = ["test erreur"] * 10
        r1 = bootstrap_cer(hyp, ref, n_iter=100, seed=42)
        r2 = bootstrap_cer(hyp, ref, n_iter=100, seed=42)
        assert r1["cer_moyen"] == r2["cer_moyen"]


class TestMcNemar:
    def test_pas_de_difference(self):
        erreurs = [True, False, True, False]
        res = comparer_mcnemar(erreurs, erreurs)
        assert res["statistique"] == 0.0

    def test_difference_significative(self):
        a = [True] * 50 + [False] * 50
        b = [False] * 100
        res = comparer_mcnemar(a, b)
        assert res["significatif"] is True

    def test_structure_retour(self):
        res = comparer_mcnemar([True, False], [False, True])
        for cle in ["b", "c", "statistique", "p_value", "significatif"]:
            assert cle in res


class TestNeedsReview:
    def test_aucune_revision(self):
        resultats = [{"needs_review": False}] * 5
        assert taux_needs_review(resultats) == 0.0

    def test_toutes_revision(self):
        resultats = [{"needs_review": True}] * 4
        assert taux_needs_review(resultats) == 1.0

    def test_liste_vide(self):
        assert taux_needs_review([]) == 0.0
