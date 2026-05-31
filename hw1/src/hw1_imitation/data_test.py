from hw1_imitation.data import build_valid_indices
import numpy as np


def test_build_valid_indices():
    """
    episode 0: indices 0, 1, 2, 3, 4
    valid chunk starts with chunk_size=3:
    0 -> [0, 1, 2]
    1 -> [1, 2, 3]
    2 -> [2, 3, 4]

    episode 1: indices 5, 6, 7, 8, 9
    valid chunk starts:
    5 -> [5, 6, 7]
    6 -> [6, 7, 8]
    7 -> [7, 8, 9]
    """
    episode_ends = np.array([5, 10])
    chunk_size = 3

    valid_indices = build_valid_indices(episode_ends, chunk_size)

    np.testing.assert_array_equal(
        valid_indices,
        np.array([0, 1, 2, 5, 6, 7]),
    )
