import torch

from hw4.train import compute_group_advantages


def test_compute_group_advantages():
    """
    test with 2 prompt groups:
    group 0: [1, 2, 3] -> mean = 2, std = sqrt(2/3) => adv = [-1.2, 0, 1.2]
    group 1: [10, 10, 10] -> mean = 0, std = 0 => adv = [0, 0, 0]
    """
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    group_size = 3

    actual = compute_group_advantages(rewards, group_size)
    expected = torch.tensor([-1.2247449, 0.0, 1.2247449, 0.0, 0.0, 0.0])

    torch.testing.assert_close(actual=actual, expected=expected)
