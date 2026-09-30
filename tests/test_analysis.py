"""Numerical and statistical checks against independent expected behavior."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import fit_distances, h1_test, h2_test, holm, load_trajectories, rank_groups
from src.utils import ROOT, align_curves, choose_rank, delay_embed, read_scalars, read_selection


def test_episode_embeddings_do_not_cross_boundaries():
    """Separate episodes never introduce a spurious 100→0 transition."""
    first = np.arange(10)[:, None]
    second = 100 + first
    embedded = delay_embed([first, second], 2)
    assert embedded.shape == (18, 2)
    assert np.all(embedded[:, 1] - embedded[:, 0] == 1)


def test_rank_rejects_empty_constant_and_short_data():
    """Degenerate trajectories fail before producing arbitrary DSA distances."""
    for values in ([], [np.ones((20, 2))], [np.arange(4)[:, None]]):
        with pytest.raises(ValueError):
            choose_rank(values, 3)


def test_rank_energy_and_cap():
    """A known rank-one signal remains rank one even with a large rank cap."""
    time = np.linspace(0, 1, 20)
    values = np.column_stack([time, 2 * time])
    rank, supported = choose_rank([values], 1)
    assert rank == 1 and supported == [1]


def test_h2_identity_and_seeded_permutations():
    """Collinear structural and reward distances yield rho=1 reproducibly."""
    descriptors = np.arange(7.0)[:, None]
    first = h2_test(descriptors, np.arange(7.0), permutations=99, seed=17)
    second = h2_test(descriptors, np.arange(7.0), permutations=99, seed=17)
    assert first == second
    assert first["rho"] == pytest.approx(1)
    assert first["p"] <= 0.05 and first["n_pairs"] == 21
    assert np.isnan(h2_test(descriptors, np.ones(7), 9)["rho"])


def test_h1_group_distance_difference():
    """A fixture with dispersed bottom systems gives the expected mean difference."""
    matrix = np.array([[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 4], [5, 5, 4, 0.0]])
    result = h1_test(matrix, ["Top", "Top", "Bottom", "Bottom"], 99, 3)
    assert result["top_mean"] == 1 and result["bottom_mean"] == 4
    assert result["difference"] == 3 and 0 < result["p"] <= 1


def test_holm_known_values():
    """Holm adjustment matches its step-down formula and retains undefined tests."""
    np.testing.assert_allclose(holm([0.01, 0.03, 0.04, np.nan]), [0.03, 0.06, 0.06, np.nan])


def test_rank_groups_identity_and_insufficient_counts():
    """Groups never overlap and original identifiers survive performance sorting."""
    frame = pd.DataFrame({"csv_idx": [9, 4, 6, 8], "stratum": ["null"] * 4})
    scores = pd.DataFrame({"csv_idx": [9, 4, 6, 8], "performance": [1, 4, 2, 3]})
    groups = rank_groups(frame, scores, 2)
    assert set(groups.loc[groups.group == "Top", "csv_idx"]) == {4, 8}
    assert set(groups.loc[groups.group == "Bottom", "csv_idx"]) == {9, 6}
    with pytest.raises(ValueError):
        rank_groups(frame, scores, 3)


def test_actual_step_alignment():
    """Curves with different sample times are interpolated rather than zipped by index."""
    grid, mean, std = align_curves(
        [(np.array([0, 10, 20]), np.array([0, 10, 20])), (np.array([5, 15, 25]), np.array([5, 15, 25]))]
    )
    np.testing.assert_array_equal(grid, [5, 10, 15, 20])
    np.testing.assert_array_equal(mean, grid)
    np.testing.assert_array_equal(std, np.zeros(4))


def test_missing_trajectories_filter_metadata(tmp_path):
    """An allowed subset filters descriptors and arrays by the same original rows."""
    frame = read_selection(ROOT / "selected.csv").head(3)
    for _, row in frame.iloc[[0, 2]].iterrows():
        prefix = tmp_path / f"Swimmer-v4_{row.stratum}_{row.seed}"
        np.save(str(prefix) + "_Ct.npy", np.arange(20)[:, None])
        np.save(str(prefix) + "_Ot.npy", np.arange(20)[:, None])
    args = SimpleNamespace(protocol="legacy", trajectories_csv=None, trajectories_dir=tmp_path, allow_subset=False)
    with pytest.raises(FileNotFoundError):
        load_trajectories(args, frame, "Swimmer-v4")
    args.allow_subset = True
    kept, states, inputs, excluded = load_trajectories(args, frame, "Swimmer-v4")
    assert list(kept.csv_idx) == [0, 2] and len(states) == len(inputs) == 2
    assert excluded[0]["csv_idx"] == 1


def test_tensorboard_preserves_all_scalars(tmp_path):
    """Event loading does not retain only TensorBoard's default 10k reservoir sample."""
    from src.utils import configure_tensorboard

    configure_tensorboard()
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path))
    for step in range(10050):
        writer.add_scalar("rollout/ep_rew_mean", float(step), step)
    writer.close()
    steps, values = read_scalars(tmp_path)
    assert len(steps) == 10050
    np.testing.assert_array_equal(steps, values)


@pytest.mark.integration
def test_real_dmdc_backend_ragged_episodes():
    """Corrected DSA uses true DMDc with independent episodes of differing lengths."""
    rng = np.random.default_rng(9)
    xs, us = [], []
    for gain in (1.0, 1.1):
        episodes, controls = [], []
        for length in (50, 65):
            control = rng.normal(size=(length, 1))
            state = np.zeros((length, 2))
            for time in range(length - 1):
                state[time + 1] = [
                    0.7 * state[time, 0] + gain * control[time, 0],
                    0.3 * state[time, 1] + 0.2 * state[time, 0],
                ]
            episodes.append(state)
            controls.append(control)
        xs.append(episodes)
        us.append(controls)
    args = SimpleNamespace(
        protocol="corrected",
        backend="dmdc",
        n_delays=1,
        rank=2,
        rank_energy=0.99,
        min_rank=1,
        max_rank=5,
        dmd_regularization=1e-8,
        analysis_seed=0,
        workers=1,
        device="cpu",
        state_metric="wasserstein",
        state_iters=10,
        state_learning_rate=1e-3,
    )
    matrices, diagnostics = fit_distances(xs, us, args)
    assert diagnostics["dmd_class"] == "DMDc"
    assert len(matrices) == 5
    for matrix in matrices.values():
        assert matrix.shape == (2, 2) and np.isfinite(matrix).all()
