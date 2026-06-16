import random
import numpy as np


def fixer_seeds(seed: int = 42) -> None:
    """Fixe tous les seeds pour la reproductibilite.

    Args:
        seed: Valeur du seed (defaut : 42).

    Example:
        >>> fixer_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seeds fixes a {seed}")
