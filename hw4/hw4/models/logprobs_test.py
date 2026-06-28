from collections import namedtuple

import torch

from hw4.models.logprobs import compute_per_token_logprobs

FixedLogitsModelOutput = namedtuple("FixedLogitsModelOutput", ["logits"])


class FixedLogitsModel(torch.nn.Module):
    """A dummy model that returns fixed pre-seeded logits."""

    def __init__(self, logits: torch.Tensor):
        super(FixedLogitsModel, self).__init__()
        self.logits = logits

    def forward(self, *args, **kwargs):
        return FixedLogitsModelOutput(logits=self.logits)


def test_shape_compute_per_token_logprobs():
    B, L, V = 32, 64, 512
    logits = torch.randn(B, L, V)
    model = FixedLogitsModel(logits)

    input_ids = torch.randint(low=0, high=V, size=(B, L))
    attention_mask = torch.ones_like(input_ids)

    out = compute_per_token_logprobs(
        model=model, input_ids=input_ids, attention_mask=attention_mask
    )

    assert out.shape == (B, L - 1)


def test_content_compute_per_token_logprobs():
    B, L, V = 1, 3, 4
    input_ids = torch.tensor([[0, 2, 1]])
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],  # used to score token 2
                [4.0, 3.0, 2.0, 1.0],  # used to score token 1
                [
                    0,
                    0,
                    0,
                    0,
                ],  # unused, final position
            ]
        ]
    )
    model = FixedLogitsModel(logits)

    all_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    """
    output[0, 0] = log p(token 2 | token 0)
                 = logprob of input_ids[0, 1]
                 = all_log_probs[batch 0, logits position 0, vocab token 2]

    output[0, 1] = log p(token 1 | tokens 0,2)
                 = logprob of input_ids[0, 2]
                 = all_log_probs[batch 0, logits position 1, vocab token 1]

    note token i means i-th token in vocab, not i-th token in sequence.

    then, expected would be:

    [
        [
            log p(x_1 | x_0),
            log p(x_2 | x_0, x_1)
        ]
    ]
    """
    expected = torch.tensor(
        [
            [
                all_log_probs[0, 0, 2],
                all_log_probs[0, 1, 1],
            ]
        ]
    )

    attention_mask = torch.ones_like(input_ids)
    out = compute_per_token_logprobs(
        model=model, input_ids=input_ids, attention_mask=attention_mask
    )

    torch.testing.assert_close(out, expected)
