"""Ranking, post-update alignment, cache safety, and CEBRA integration."""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cli import main
from src.config import parse_args
from src.context_embeddings import (
    aligned_episode,
    cached_agent,
    combine_trials,
    embed_task,
    fit_pca,
    read_evaluation,
    select_agents,
    validate_cache_manifest,
)
from src.evaluation import rollout
from src.utils import ROOT, configure_tensorboard, file_hash, matrix_hash
from tests.test_protocol import FixedPolicy, wrapper


@pytest.fixture(autouse=True)
def tensorboard_stub():
    """Use the production TensorBoard setup before importing PPO dependencies."""
    configure_tensorboard()


def embedding_args(**kwargs):
    """Use small CPU fits while exercising the real CEBRA estimator."""
    return SimpleNamespace(
        analysis_seed=7,
        cebra_iterations=2,
        cebra_batch_size=16,
        cebra_learning_rate=3e-4,
        cebra_delta=0.1,
        device="cpu",
        embedding_mode="separate",
        dpi=40,
        **kwargs,
    )


def test_individual_policy_ranking_and_disjoint_ties():
    """Different policy seeds of one reservoir rank as separate agents."""
    episodes = pd.DataFrame(
        [
            {"task": "A", "csv_idx": 2, "policy_seed": 0, "return": 9},
            {"task": "A", "csv_idx": 2, "policy_seed": 0, "return": 11},
            {"task": "A", "csv_idx": 2, "policy_seed": 1, "return": 0},
            {"task": "A", "csv_idx": 1, "policy_seed": 0, "return": 10},
        ]
    )
    ranking, selected = select_agents(episodes, 1, "both")
    assert ranking.csv_idx.tolist() == [1, 2, 2]
    assert selected.csv_idx.tolist() == [1, 2]
    assert selected.policy_seed.tolist() == [0, 1]
    assert selected.group.tolist() == ["top", "bottom"]
    episodes["return"] = 1
    _, selected = select_agents(episodes, 1, "both")
    assert not selected.duplicated(["task", "csv_idx", "policy_seed"]).any()
    with pytest.raises(ValueError, match="insufficient"):
        select_agents(episodes, 2, "both")


@pytest.mark.parametrize("protocol", ["legacy", "corrected"])
def test_collector_records_actual_actions_without_skip_features(protocol):
    """Activity excludes the skip input and labels retain executed legacy actions."""
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(wrapper(protocol=protocol, skip=True))
    try:
        result = rollout(FixedPolicy(), env, 4, collect=True)
        assert result["post_states"].shape == (3, 8)
        np.testing.assert_allclose(result["actions"], [[0.25, -0.5]] * 3)
        np.testing.assert_allclose(result["rewards"], [1] * 3)
        np.testing.assert_allclose(result["observations"], [[1, 0.1], [1, 0.2], [1, 0.3]])
        if protocol == "legacy":
            np.testing.assert_array_equal(result["inputs"][:, 2:4], 0)
    finally:
        env.close()


def test_corrected_adapter_matches_post_states_and_terminal():
    """Old InputDSA arrays shift context once and drop the duplicate last input."""
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(wrapper())
    try:
        result = rollout(FixedPolicy(), env, 1, collect=True)
        old = {
            "states": np.vstack([result["states"], result["terminal_state"]]),
            "inputs": np.vstack([result["inputs"], result["inputs"][-1]]),
        }
        adapted = aligned_episode(old, "corrected", 2, 2, "continuous")
        np.testing.assert_array_equal(adapted["contexts"], result["post_states"])
        np.testing.assert_array_equal(adapted["actions"], result["actions"])
        assert len(adapted["contexts"]) == 3
        assert aligned_episode(old, "legacy", 2, 2, "continuous") is None
        trials = [{**adapted, "episode": i, "evaluation_seed": 100 + i} for i in range(2)]
        combined = combine_trials(trials)
        np.testing.assert_array_equal(combined["timesteps"], [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(combined["episodes"], [0, 0, 0, 1, 1, 1])
    finally:
        env.close()


def test_categorical_action_adapter():
    """One-hot corrected inputs become one-dimensional integer labels."""
    old = {
        "states": np.arange(20).reshape(5, 4),
        "inputs": np.column_stack([np.ones((5, 2)), np.eye(3)[[2, 0, 1, 2, 2]], np.ones(5)]),
    }
    adapted = aligned_episode(old, "corrected", 2, 3, "discrete")
    np.testing.assert_array_equal(adapted["actions"], [2, 0, 1, 2])
    assert adapted["actions"].dtype == np.int64


@pytest.mark.parametrize("fault", ["length", "nan", "dimensions"])
def test_invalid_aligned_data(fault):
    """Malformed observations cannot silently pair with reservoir samples."""
    data = {
        "contexts": np.ones((5, 4)),
        "observations": np.ones((5, 2)),
        "actions": np.ones((5, 2)),
        "rewards": np.ones(5),
    }
    if fault == "length":
        data["rewards"] = np.ones(4)
    elif fault == "nan":
        data["contexts"][0, 0] = np.nan
    else:
        data["observations"] = np.ones((5, 3))
    with pytest.raises(ValueError):
        aligned_episode(data, "corrected", 2, 2, "continuous")


def test_pca_preserves_neuron_variance():
    """PCA centers raw activity rather than standardizing neurons."""
    contexts = np.random.default_rng(4).normal(size=(30, 5)) * [100, 2, 1, 1, 1]
    model, embedding = fit_pca(contexts)
    assert embedding.shape == (30, 3)
    assert model.explained_variance_ratio_[0] > 0.99
    with pytest.raises(ValueError, match="three"):
        fit_pca(contexts[:2])
    with pytest.raises(ValueError, match="constant"):
        fit_pca(np.ones((5, 4)))


def test_cache_manifest_rejects_protocol_selection_and_seed_mismatches(tmp_path):
    """Reuse requires the same protocol, selection, and common episode sequence."""
    path = tmp_path / "evaluation" / "episodes.csv"
    path.parent.mkdir()
    args = SimpleNamespace(
        selected_csv=ROOT / "selected.csv",
        protocol="legacy",
        evaluation_seed=10000,
        evaluation_episodes=10,
        max_episode_steps=None,
    )
    metadata = {
        "selection_sha256": file_hash(args.selected_csv),
        "status": "completed",
        "arguments": {"protocol": "legacy", "evaluation_seed": 10000, "evaluation_episodes": 10},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    validate_cache_manifest(path, args)
    for key, wrong in (("protocol", "corrected"), ("evaluation_seed", 3), ("evaluation_episodes", 2)):
        invalid = {**metadata, "arguments": {**metadata["arguments"], key: wrong}}
        (tmp_path / "manifest.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="mismatch"):
            validate_cache_manifest(path, args)
    metadata["selection_sha256"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_cache_manifest(path, args)


def test_evaluation_cache_checks_checkpoint_hash(tmp_path):
    """Cached scores for a replaced checkpoint are rejected."""
    model = tmp_path / "model.zip"
    model.write_bytes(b"checkpoint")
    path = tmp_path / "episodes.csv"
    args = SimpleNamespace(
        protocol="legacy", selected_csv=ROOT / "selected.csv", evaluation_seed=10000, evaluation_episodes=1
    )
    record = {"task": "A", "csv_idx": 0, "policy_seed": 0, "reservoir_seed": 42, "model_path": str(model)}
    pd.DataFrame(
        [
            {
                **record,
                "evaluation_seed": 10000,
                "return": 10,
                "protocol": "legacy",
                "selection_sha256": file_hash(args.selected_csv),
                "model_sha256": "wrong",
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="checkpoint"):
        read_evaluation(path, args, pd.DataFrame([record]))


@pytest.mark.parametrize("case", ["modern", "old_legacy", "missing", "wrong_matrix"])
def test_cached_agent_reuse_or_recollection(tmp_path, case):
    """Reuse complete labeled trials and request recollection for missing labels."""
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(wrapper(protocol="legacy"))
    result = rollout(FixedPolicy(), env, 10000, collect=True)
    model = tmp_path / "model.zip"
    model.write_bytes(b"checkpoint")
    identity = {
        "task": "A",
        "csv_idx": 0,
        "policy_seed": 0,
        "reservoir_seed": 42,
        "episode": 0,
        "evaluation_seed": 10000,
        "protocol": "legacy",
        "selection_sha256": "selection",
    }
    record = pd.Series({**identity, "model_path": str(model), "encoder_sha256": "encoder"})
    path = tmp_path / "episode.npz"
    payload = {**identity, "states": result["post_states"], "inputs": result["inputs"]}
    if case != "old_legacy":
        payload.update(
            contexts=result["post_states"],
            observations=result["observations"],
            actions=result["actions"],
            rewards=result["rewards"],
        )
    if case != "missing":
        np.savez_compressed(path, **payload)
    registry = pd.DataFrame(
        [
            {
                **identity,
                "path": str(path),
                "matrix_hash": "wrong" if case == "wrong_matrix" else matrix_hash(env.env.reservoir.W),
            }
        ]
    )
    args = SimpleNamespace(
        protocol="legacy", evaluation_seed=10000, evaluation_episodes=1, trajectories_csv=tmp_path / "trajectories.csv"
    )
    try:
        if case == "wrong_matrix":
            with pytest.raises(ValueError, match="matrix_hash mismatch"):
                cached_agent(args, record, registry, env)
        else:
            trials = cached_agent(args, record, registry, env)
            if case == "modern":
                assert len(trials) == 1
                np.testing.assert_array_equal(trials[0]["actions"], result["actions"])
            else:
                assert trials is None
    finally:
        env.close()


def test_embedding_cli_defaults_and_dry_run(tmp_path):
    """New script defaults to all five tasks and dry-run never creates outputs."""
    args = parse_args("plot_context_embeddings", ["--protocol", "legacy"])
    assert len(args.tasks) == 5 and args.k == 1 and args.group == "top"
    assert args.cebra_iterations == 10000
    # Model discovery is mocked so this check does not require historical assets.
    from unittest.mock import patch

    registry = pd.DataFrame([{"task": "Swimmer-v4", "csv_idx": 0, "policy_seed": 0}])
    with (
        patch("src.context_embeddings.load_registry", return_value=registry),
        patch("src.context_embeddings.registry_environment", return_value=({}, None)),
    ):
        main(
            "plot_context_embeddings",
            ["--protocol", "legacy", "--tasks", "Swimmer-v4", "--dry-run", "--output-dir", str(tmp_path / "output")],
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.integration
def test_short_cebra_continuous_discrete_and_constant(tmp_path):
    """Real CPU models train, transform, save, and reload on both label types."""
    # Other repository tests load TensorFlow/POT's native libraries. Keep the
    # real estimator in a fresh process, matching this script's CLI lifecycle,
    # to avoid a native LLVM collision when Torch lazily loads Triton.
    code = textwrap.dedent("""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        import torch
        import cebra
        from src.context_embeddings import fit_cebra

        torch.set_num_threads(1)
        contexts = np.random.default_rng(2).normal(size=(32, 6)).astype(np.float32)
        args = SimpleNamespace(analysis_seed=7, cebra_iterations=2, cebra_batch_size=16,
                               cebra_learning_rate=3e-4, cebra_delta=0.1, device="cpu")
        for discrete, labels in ((False, contexts[:, :2]), (True, np.arange(32) % 3)):
            model, embedding, scaler = fit_cebra(contexts, labels, discrete, args)
            assert embedding.shape == (32, 3)
            assert (scaler is None) == discrete
            path = Path(sys.argv[1]) / f"cebra_{discrete}.pt"
            model.save(str(path))
            restored = cebra.CEBRA.load(str(path), weights_only=False)
            np.testing.assert_allclose(restored.transform(contexts), embedding)
        assert fit_cebra(contexts, np.ones(32), False, args) == (None, None, None)
    """)
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60
    )


@pytest.mark.parametrize("mode", ["separate", "shared"])
def test_embedding_panels_and_saved_shapes(tmp_path, mode, monkeypatch):
    """Separate mode fits agents independently; shared mode pools within a task."""
    rng = np.random.default_rng(3)
    args = embedding_args()
    args.embedding_mode = mode
    for folder in ("models", "analysis", "figures"):
        (tmp_path / folder).mkdir()
    agents = []
    for index in range(2):
        agents.append(
            {
                "record": pd.Series(
                    {
                        "task": "A",
                        "csv_idx": index,
                        "policy_seed": 0,
                        "group": "top",
                        "rank": index + 1,
                        "mean_return": 10,
                    }
                ),
                "data": {
                    "contexts": rng.normal(size=(10, 4)),
                    "observations": rng.normal(size=(10, 2)),
                    "actions": rng.normal(size=(10, 2)),
                    "rewards": np.ones(10),
                    "episodes": np.zeros(10),
                    "evaluation_seeds": np.full(10, 10000),
                    "timesteps": np.arange(10),
                    "action_type": "continuous",
                },
            }
        )
    # Actual fitting is covered above; this exercises independent plotting/export.
    monkeypatch.setattr("src.context_embeddings.fit_cebra", lambda *a: (None, None, None))
    embed_task(args, "A", agents, tmp_path)
    files = sorted((tmp_path / "analysis").glob("*.npz"))
    assert len(files) == (2 if mode == "separate" else 1)
    for path in files:
        with np.load(path) as data:
            assert data["pca"].shape == (10 if mode == "separate" else 20, 3)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
