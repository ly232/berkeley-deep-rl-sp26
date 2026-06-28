import torch

from hw4.rollout.rollout_buffer import iter_minibatches, RolloutBatch


def make_rollout_batch(test_data):
    return RolloutBatch(
        input_ids=test_data,
        attention_mask=test_data,
        completion_mask=test_data,
        old_logprobs=test_data,
        ref_logprobs=test_data,
        rewards=test_data,
        advantages=test_data,
    )


def test_iter_minibatches():
    test_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    # Note many of these data do not match semantics, e.g. probability cannot
    # be larger than 1. But fake data suffices for testing iter_minibatches.
    rollout_batch = make_rollout_batch(test_data)
    rollout_batch.task_names = ["task 1", "task 2"]
    rollout_batch.completion_texts = ["hello", "world"]
    actual = [
        elem
        for elem in iter_minibatches(
            batch=rollout_batch, minibatch_size=1, shuffle=False
        )
    ]

    torch.testing.assert_close(actual[0].input_ids, test_data[0].unsqueeze(0))
    torch.testing.assert_close(actual[1].input_ids, test_data[1].unsqueeze(0))

    assert actual[0].task_names == ["task 1"]
    assert actual[1].task_names == ["task 2"]

    assert actual[0].completion_texts == ["hello"]
    assert actual[1].completion_texts == ["world"]
