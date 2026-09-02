import numpy as np

from fastweights import FastWeightMemory, cosine, make_problem, make_vectors


def test_single_association_has_perfect_recall():
    key = np.array([1.0, 0.0, 0.0])
    value = np.array([0.0, 1.0, 0.0])
    memory = FastWeightMemory(dimension=3)

    memory.write(key, value)

    np.testing.assert_allclose(memory.read(key), value)
    assert memory.top1_accuracy() == 1.0


def test_overlapping_keys_degrade_cosine_similarity_monotonically():
    dimension = 8
    key = np.zeros(dimension)
    key[0] = 1.0
    similarities = []

    for association_count in range(1, 6):
        memory = FastWeightMemory(dimension=dimension)
        for index in range(association_count):
            value = np.zeros(dimension)
            value[index] = 1.0
            memory.write(key, value)
        estimate = memory.read(key)
        similarities.append(estimate[0] / np.linalg.norm(estimate))

    assert all(left > right for left, right in zip(similarities, similarities[1:]))


def test_orthogonal_keys_have_no_crosstalk():
    dimension = 4
    memory = FastWeightMemory(dimension=dimension)

    for index in range(dimension):
        key = np.eye(dimension)[index]
        value = np.roll(key, 1)
        memory.write(key, value)

    for index in range(dimension):
        estimate, signal, crosstalk = memory.decompose_read(index)
        np.testing.assert_allclose(signal, memory.values[index])
        np.testing.assert_allclose(crosstalk, 0.0)
        np.testing.assert_allclose(estimate, memory.values[index])

    assert memory.top1_accuracy() == 1.0


def test_correlated_generator_matches_requested_mean_cosine():
    for rho in (0.0, 0.3, 0.6, 0.9):
        vectors = make_vectors(n=200, d=256, rho=rho, seed=123)
        similarities = vectors @ vectors.T
        off_diagonal = similarities[~np.eye(len(vectors), dtype=bool)]
        assert abs(off_diagonal.mean() - rho) < 0.03
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_cosine_matches_unit_vector_dot_product():
    first = np.array([1.0, 0.0])
    second = np.array([1.0, 1.0]) / np.sqrt(2.0)

    assert np.isclose(cosine(first, second), 1.0 / np.sqrt(2.0))


def test_decomposition_matches_matrix_read_with_and_without_decay():
    keys = make_vectors(n=6, d=16, rho=0.4, seed=7)
    values = make_vectors(n=6, d=16, rho=0.0, seed=8)
    for decay in (1.0, 0.7):
        memory = FastWeightMemory.from_arrays(keys, values, decay=decay)
        for index in range(len(keys)):
            estimate, signal, crosstalk = memory.decompose_read(index)
            np.testing.assert_allclose(estimate, memory.read(memory.keys[index]))
            np.testing.assert_allclose(estimate, signal + crosstalk)


def test_incremental_and_batched_construction_match():
    keys = make_vectors(n=10, d=32, rho=0.2, seed=9)
    values = make_vectors(n=10, d=32, rho=0.0, seed=10)
    incremental = FastWeightMemory(dimension=32, decay=0.8)
    for key, value in zip(keys, values):
        incremental.write(key, value)

    batched = FastWeightMemory.from_arrays(keys, values, decay=0.8)

    np.testing.assert_allclose(incremental.matrix, batched.matrix)
    np.testing.assert_allclose(
        incremental.read(keys[3]), batched.read(keys[3])
    )
    assert incremental.top1_accuracy() == batched.top1_accuracy()


def test_make_problem_isolates_key_correlation_in_accuracy_sweep():
    accuracies = []
    value_correlations = []
    n = 40
    for rho in np.linspace(0.0, 0.9, 10):
        keys, values = make_problem(n=n, d=64, rho=float(rho), seed=123)
        accuracies.append(FastWeightMemory.from_arrays(keys, values).top1_accuracy())
        gram = values @ values.T
        off_diagonal = gram[~np.eye(n, dtype=bool)]
        value_correlations.append(off_diagonal.mean())

    assert all(left >= right for left, right in zip(accuracies, accuracies[1:]))
    np.testing.assert_allclose(value_correlations, 0.0, atol=0.05)