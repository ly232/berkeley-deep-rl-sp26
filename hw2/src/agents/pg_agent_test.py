from agents.pg_agent import PGAgent
import numpy as np
import pytest


@pytest.mark.parametrize("use_reward_to_go", [False, True])
def test_calculate_q_vals(use_reward_to_go):
    # All args are irrelevant except for gamma.
    agent = PGAgent(
        ob_dim=0,
        ac_dim=0,
        discrete=False,
        n_layers=0,
        layer_size=0,
        gamma=0.9,
        learning_rate=0,
        use_baseline=False,
        use_reward_to_go=use_reward_to_go,
        baseline_learning_rate=None,
        baseline_gradient_steps=None,
        gae_lambda=None,
        normalize_advantages=False,
    )

    q_vals = agent._calculate_q_vals(
        [
            [10, -5, 20],
            [0, 0, 1],
        ]
    )
    if use_reward_to_go:
        np.testing.assert_array_equal(
            q_vals[0],
            np.array([10 + 0.9 * (-5) + 0.9**2 * 20, -5 + 0.9 * 20, 20]),
        )
        np.testing.assert_array_equal(
            q_vals[1],
            np.array([0 + 0.9 * 0 + 0.9**2 * 1, 0 + 0.9 * 1, 1]),
        )
    else:
        np.testing.assert_array_equal(
            q_vals[0],
            np.array([10 + 0.9 * (-5) + 0.9**2 * 20] * 3),
        )
        np.testing.assert_array_equal(
            q_vals[1],
            np.array([0 + 0.9 * 0 + 0.9**2 * 1] * 3),
        )
