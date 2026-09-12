"""One forward and one backward pass of the two-head net."""

from __future__ import annotations

import esca.tensors as tensors
import pytest
import torch
from torch.nn import functional as F

from pyanglerfish import DEFAULT_GROUPS, NetConfig, TwoHeadNet
from pyanglerfish.features import TORCH_DTYPES
from pyanglerfish.moves import MOVE_DTYPES, MOVE_FIELDS

ROWS = 3
MOVES = 5
FEATURES = ("state.in_check", "material.pawns", "placement.pawns", "history.halfmove_clock", "king.tropism")


def tiny() -> tuple[TwoHeadNet, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """A net over `FEATURES` and one batch of typed arrays for it."""
    torch.manual_seed(0)
    net = TwoHeadNet(NetConfig(features=FEATURES, trunk=(16,), embedding=8, policy_hidden=4))
    known = {entry["name"]: entry for entry in tensors.layout()}
    facts = {
        name: torch.randint(0, 2, (ROWS, *known[name]["shape"])).to(TORCH_DTYPES[known[name]["dtype"]])
        for name in FEATURES
    }
    moves = {field: torch.randint(0, 2, (ROWS, MOVES)).to(TORCH_DTYPES[MOVE_DTYPES[field]]) for field in MOVE_FIELDS}
    mask = torch.ones(ROWS, MOVES, dtype=torch.bool)
    mask[0, 3:] = False
    mask[1, 1:] = False
    return net, facts, moves, mask


def test_the_input_width_is_the_layouts() -> None:
    config = NetConfig(features=FEATURES)
    # in_check 1, pawns 2, the pawn planes 2·64, the clock 1, tropism 2
    assert config.input_width == 134
    assert config.move_width == len(MOVE_FIELDS)


def test_the_default_selection_is_every_group_but_maps() -> None:
    config = NetConfig()
    assert set(config.features) == set(tensors.names(list(DEFAULT_GROUPS)))
    assert not any(name.startswith("maps.") for name in config.features)


def test_forward_shapes_and_masking() -> None:
    net, facts, moves, mask = tiny()
    value, policy = net(facts, moves, mask)
    assert value.shape == (ROWS,)
    assert policy.shape == (ROWS, MOVES)
    assert torch.all(torch.isfinite(value))
    assert torch.all(torch.isinf(policy[~mask]))
    assert torch.all(torch.isfinite(policy[mask]))
    priors = policy.softmax(dim=1)
    assert torch.allclose(priors.sum(dim=1), torch.ones(ROWS), atol=1e-5)
    assert torch.all(priors[~mask] == 0.0)


def test_backward_reaches_every_parameter() -> None:
    net, facts, moves, mask = tiny()
    value, policy = net(facts, moves, mask)
    loss = F.binary_cross_entropy_with_logits(value, torch.tensor([0.9, 0.1, 0.5])) + F.cross_entropy(
        policy, torch.tensor([2, 0, 4])
    )
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.all(torch.isfinite(parameter.grad)), name


def test_a_step_lowers_the_loss_on_one_batch() -> None:
    net, facts, moves, mask = tiny()

    def loss_now() -> torch.Tensor:
        value, policy = net(facts, moves, mask)
        return F.binary_cross_entropy_with_logits(value, torch.tensor([0.9, 0.1, 0.5])) + F.cross_entropy(
            policy, torch.tensor([2, 0, 4])
        )

    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-2)
    before = float(loss_now())
    for _ in range(20):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_now()
        loss.backward()
        optimiser.step()
    assert float(loss_now()) < before


def test_an_array_the_batch_lacks_is_an_error() -> None:
    net, facts, moves, mask = tiny()
    with pytest.raises(KeyError):
        net({name: facts[name] for name in FEATURES[:2]}, moves, mask)
