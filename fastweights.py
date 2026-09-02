"""Dense Hebbian fast-weight associative memory."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity, rejecting zero or non-finite vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be finite vectors with the same shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must contain only finite values")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / denominator)


def make_vectors(
    n: int, d: int, rho: float = 0.0, seed: int | None = None
) -> np.ndarray:
    """Generate ``n`` unit vectors in ``R^d`` with mean overlap near ``rho``.

    Draw a unit common direction ``c`` and independent Gaussian noise vectors,
    project each noise vector into the subspace perpendicular to ``c``, and
    normalize them to unit vectors ``u_i``. The returned vectors are
    ``k_i = sqrt(rho) * c + sqrt(1 - rho) * u_i``. Since ``c`` is perpendicular
    to every ``u_i``, each ``k_i`` is exactly unit-normalized. For distinct
    items, ``E[k_i . k_j] = rho`` because ``E[u_i . u_j] = 0``; finite samples
    therefore have an empirical mean pairwise cosine approximately equal to
    ``rho``. At ``rho=0``, the vectors span the ``(d - 1)``-dimensional
    subspace perpendicular to ``c``, so the crosstalk approximation is
    ``sqrt(N / (d - 1))`` rather than ``sqrt(N / d)``. This difference is
    negligible at large ``d`` but visible at ``d=16``. This is an
    equicorrelation construction: all overlap comes from one shared direction,
    so its expected Gram matrix is rank-one-plus-identity; learned keys can
    have a richer correlation structure. ``rho`` must be in ``[0, 1)`` and
    ``d`` must be at least two.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if d < 2:
        raise ValueError("d must be at least 2")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in the interval [0, 1)")

    generator = np.random.default_rng(seed)
    common = generator.normal(size=d)
    common /= np.linalg.norm(common)
    noise = generator.normal(size=(n, d))
    noise -= (noise @ common)[:, None] * common
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    vectors = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * noise
    return vectors


def make_problem(
    n: int, d: int, rho: float = 0.0, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate keys and independent values for one memory experiment.

    Keys use the requested ``rho`` while values always use ``rho=0``. The two
    calls receive different child seeds derived from ``seed``. This matters
    because ``top1_accuracy`` ranks retrieved vectors against stored values;
    correlated values would independently make that ranking harder and would
    confound the experiment intended to isolate key interference.
    """
    seed_sequence = np.random.SeedSequence(seed)
    key_sequence, value_sequence = seed_sequence.spawn(2)
    key_seed = int(key_sequence.generate_state(1)[0])
    value_seed = int(value_sequence.generate_state(1)[0])
    keys = make_vectors(n=n, d=d, rho=rho, seed=key_seed)
    values = make_vectors(n=n, d=d, rho=0.0, seed=value_seed)
    return keys, values


class FastWeightMemory:
    """A fixed-size associative memory updated with outer-product writes.

    Keys and values are required to be unit-normalized. Each write applies
    ``M <- decay * M + value @ key.T`` and each read applies ``M @ query``.
    ``matrix`` is the memory mechanism: it remains ``d x d`` regardless of
    how many associations are written, and reads use it alone. ``keys`` and
    ``values`` retain the associations only as teaching scaffolding for
    ground-truth comparisons and decomposition; they play no role in reads.
    """

    def __init__(self, dimension: int, decay: float = 1.0) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in the interval (0, 1]")

        self.dimension = dimension
        self.decay = decay
        self.matrix = np.zeros((dimension, dimension), dtype=float)
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    @classmethod
    def from_arrays(
        cls, keys: np.ndarray, values: np.ndarray, decay: float = 1.0
    ) -> "FastWeightMemory":
        """Build a memory from ``(N, d)`` key and value arrays in one batch."""
        keys = np.asarray(keys, dtype=float)
        values = np.asarray(values, dtype=float)
        if keys.ndim != 2 or values.shape != keys.shape:
            raise ValueError("keys and values must have the same shape (N, d)")
        if keys.shape[0] < 1:
            raise ValueError("at least one association is required")

        memory = cls(dimension=keys.shape[1], decay=decay)
        for name, vectors in (("keys", keys), ("values", values)):
            if not np.all(np.isfinite(vectors)):
                raise ValueError(f"{name} must contain only finite values")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0):
                raise ValueError(f"every {name[:-1]} must be unit-normalized")

        weights = decay ** np.arange(keys.shape[0] - 1, -1, -1)
        memory.matrix = (values * weights[:, None]).T @ keys
        memory.keys = [key.copy() for key in keys]
        memory.values = [value.copy() for value in values]
        return memory

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Store one unit-normalized key-value association."""
        key = self._validate_vector(key, "key")
        value = self._validate_vector(value, "value")
        self.matrix = self.decay * self.matrix + np.outer(value, key)
        self.keys.append(key.copy())
        self.values.append(value.copy())

    def read(self, query: np.ndarray) -> np.ndarray:
        """Read the value estimate produced by the current memory matrix."""
        query = self._validate_vector(query, "query")
        return self.matrix @ query

    def decompose_read(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(estimate, signal, crosstalk)`` for a stored key.

        For decay equal to one, this is the exact expansion of ``M @ k_j``.
        With decay, each stored contribution is weighted by the decay applied
        after it was written, so the returned terms still sum exactly to the
        current read while the target signal need not remain one.
        """
        if not 0 <= index < len(self.keys):
            raise IndexError("stored association index out of range")

        query = self.keys[index]
        signal = np.zeros(self.dimension, dtype=float)
        crosstalk = np.zeros(self.dimension, dtype=float)
        association_count = len(self.keys)
        for stored_index, (key, value) in enumerate(zip(self.keys, self.values)):
            contribution = self.decay ** (association_count - 1 - stored_index)
            term = contribution * value * np.dot(key, query)
            if stored_index == index:
                signal += term
            else:
                crosstalk += term
        estimate = signal + crosstalk
        return estimate, signal, crosstalk

    def top1_accuracy(self) -> float:
        """Return retrieval accuracy over all stored keys.

        A retrieved vector is matched to the stored value with the largest
        dot product. Values are unit-normalized, so this is cosine ranking.
        """
        if not self.values:
            return float("nan")

        keys = np.stack(self.keys)
        values = np.stack(self.values)
        retrieved = keys @ self.matrix.T
        scores = retrieved @ values.T
        return float((scores.argmax(axis=1) == np.arange(len(keys))).mean())

    def _validate_vector(self, vector: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        if not np.isclose(np.linalg.norm(vector), 1.0):
            raise ValueError(f"{name} must be unit-normalized")
        return vector