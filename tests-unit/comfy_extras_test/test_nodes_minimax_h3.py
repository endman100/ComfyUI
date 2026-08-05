import torch

from comfy.cli_args import args as cli_args


if not torch.cuda.is_available():
    cli_args.cpu = True


from comfy_extras.nodes_minimax_h3 import MiniMaxH3TemporalRampMask


def make_mask(**overrides):
    inputs = {
        "width": 4,
        "height": 3,
        "length": 124,
        "fps": 24.0,
        "protect_seconds": 1.0,
    }
    inputs.update(overrides)
    return MiniMaxH3TemporalRampMask.execute(**inputs)[0]


def make_output(**overrides):
    inputs = {
        "width": 4,
        "height": 3,
        "length": 124,
        "fps": 24.0,
        "protect_seconds": 1.0,
    }
    inputs.update(overrides)
    return MiniMaxH3TemporalRampMask.execute(**inputs)


def test_temporal_ramp_mask_matches_one_second_transition():
    mask = make_mask()

    assert mask.shape == (124, 3, 4)
    assert torch.count_nonzero(mask[0]) == 0
    assert torch.allclose(mask[12], torch.full((3, 4), 0.5))
    assert torch.count_nonzero(mask[24] != 1.0) == 0
    assert torch.count_nonzero(mask[25:] != 1.0) == 0


def test_temporal_ramp_mask_uses_fps_and_protected_seconds():
    mask = make_mask(length=70, fps=30.0, protect_seconds=2.0)

    assert torch.allclose(mask[30], torch.full((3, 4), 0.5))
    assert torch.count_nonzero(mask[60] != 1.0) == 0


def test_zero_second_transition_still_protects_frame_zero():
    mask = make_mask(length=3, protect_seconds=0.0)

    assert torch.count_nonzero(mask[0]) == 0
    assert torch.count_nonzero(mask[1:] != 1.0) == 0


def test_outputs_loader_frame_counts_from_same_setting():
    output = make_output()

    assert output[1] == 24
    assert output[2] == 100
