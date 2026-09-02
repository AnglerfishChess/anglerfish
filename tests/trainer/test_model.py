"""One forward and one backward pass of the two-head net."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from pyanglerfish import NetConfig, TwoHeadNet

ROWS = 3
MOVES = 5
WIDTH = 64


def tiny() -> tuple[TwoHeadNet, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    net = TwoHeadNet(NetConfig(input_width=WIDTH, trunk=(16,), embedding=8, policy_hidden=4))
    mask = torch.ones(ROWS, MOVES, dtype=torch.bool)
    mask[0, 3:] = False
    mask[1, 1:] = False
    batch = {
        "features": torch.randn(ROWS, WIDTH),
        "moves": torch.randn(ROWS, MOVES, 24),
        "move_mask": mask,
        "best": torch.tensor([2, 0, 4]),
        "value": torch.tensor([0.9, 0.1, 0.5]),
    }
    return net, batch


def test_forward_shapes_and_masking() -> None:
    net, batch = tiny()
    value, policy = net(batch["features"], batch["moves"], batch["move_mask"])
    assert value.shape == (ROWS,)
    assert policy.shape == (ROWS, MOVES)
    assert torch.all(torch.isfinite(value))
    assert torch.all(torch.isinf(policy[~batch["move_mask"]]))
    assert torch.all(torch.isfinite(policy[batch["move_mask"]]))
    priors = policy.softmax(dim=1)
    assert torch.allclose(priors.sum(dim=1), torch.ones(ROWS), atol=1e-5)
    assert torch.all(priors[~batch["move_mask"]] == 0.0)


def test_backward_reaches_every_parameter() -> None:
    net, batch = tiny()
    value, policy = net(batch["features"], batch["moves"], batch["move_mask"])
    loss = F.binary_cross_entropy_with_logits(value, batch["value"]) + F.cross_entropy(policy, batch["best"])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.all(torch.isfinite(parameter.grad)), name


def test_a_step_lowers_the_loss_on_one_batch() -> None:
    net, batch = tiny()

    def loss_now() -> torch.Tensor:
        value, policy = net(batch["features"], batch["moves"], batch["move_mask"])
        return F.binary_cross_entropy_with_logits(value, batch["value"]) + F.cross_entropy(policy, batch["best"])

    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-2)
    before = float(loss_now())
    for _ in range(20):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_now()
        loss.backward()
        optimiser.step()
    assert float(loss_now()) < before
