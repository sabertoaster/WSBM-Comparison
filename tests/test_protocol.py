"""Matrix identity, action encoding, masks, and episode boundaries."""

import gymnasium as gym
import numpy as np
import pytest

from src.er_mrl.wrappers import DeletedVelocityWrapper, ReservoirWrapper
from src.evaluation import rollout
from src.reservoirs import build_corrected, fixed_density, load_reservoir, save_reservoir
from src.selection import descriptors
from src.utils import TASKS, legacy_config, matrix_hash


class ToyEnvironment(gym.Env):
    """Small deterministic environment supporting continuous or discrete actions."""

    def __init__(self, discrete=False):
        """Configure two observation features and a three-step episode."""
        self.observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(3) if discrete else gym.spaces.Box(-1, 1, (2,), dtype=np.float64)
        self.counter = 0

    def reset(self, seed=None, options=None):
        """Return a reproducible initial observation."""
        super().reset(seed=seed)
        self.counter = 0
        return np.array([1.0, 0.5]), {}

    def step(self, action):
        """Advance one step, truncate at three, and return a unit reward."""
        self.counter += 1
        return np.array([1.0, self.counter / 10]), 1.0, False, self.counter == 3, {}


class FixedPolicy:
    """Deterministic toy policy for boundary and alignment checks."""

    def predict(self, observation, deterministic=True):
        """Return a fixed continuous action and no recurrent policy state."""
        return np.array([0.25, -0.5]), None


def wrapper(protocol="corrected", discrete=False, skip=False):
    """Build a seeded small reservoir wrapper without any simulator dependency."""
    return ReservoirWrapper(
        ToyEnvironment(discrete),
        units=8,
        lr=0.1,
        sr=0.9,
        iss=1.0,
        seed=42,
        protocol=protocol,
        reset_res=protocol == "corrected",
        skip_c=skip,
    )


def test_legacy_parameter_mapping(parameter_row):
    """Preserve the historical half-contrast and scalar topology conversion."""
    config = legacy_config(parameter_row)
    assert config["hi"] == 6.5 and config["lo"] == 3.5
    assert config["sigma"] == 10 and config["connectivity"] == 0.3
    assert config["p_negative"] == 0.1


def test_density_exact_for_unequal_sizes():
    """Expected edge count stays fixed under unequal planted block sizes."""
    assignments = np.array([0] * 3 + [1] * 7 + [2] * 10)
    probability = fixed_density(4.0, 0.2, assignments)
    weights = probability[np.ix_(assignments, assignments)]
    expected = weights[np.triu_indices(20, 1)].mean()
    assert expected == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Infeasible"):
        fixed_density(100.0, 0.8, assignments)


@pytest.mark.parametrize("motif", ["assortative", "disassortative", "core_periphery", "mixed", "null"])
def test_saved_matrix_matches_runtime(parameter_row, motif, tmp_path):
    """Each motif uses the same W across persisted artifacts and runtime input sizes."""
    row = dict(parameter_row, stratum=motif)
    if motif == "null":
        row.update(contrast_u=0, rho=1, sigma_ratio=1)
    elif motif == "disassortative":
        row["contrast_u"] = -3
    artifact = build_corrected(row, units=20)
    target = tmp_path / "matrix.npz"
    save_reservoir(target, artifact)
    loaded = load_reservoir(target, row, 20, 0.9)
    assert matrix_hash(loaded["W"]) == str(artifact["matrix_hash"])
    assert np.max(np.abs(np.linalg.eigvalsh(loaded["W"]))) == pytest.approx(0.9)
    for seed in (0, 4):
        np.random.seed(seed)
        env = ReservoirWrapper(
            ToyEnvironment(),
            units=20,
            lr=0.1,
            sr=0.9,
            iss=1.0,
            seed=row["seed"],
            recurrent_matrix=loaded["W"],
            protocol="corrected",
            reset_res=True,
        )
        env.reset(seed=seed)
        assert matrix_hash(env.reservoir.W) == str(loaded["matrix_hash"])
        current_readin = matrix_hash(env.reservoir.Win)
        if seed == 0:
            expected_readin = current_readin
        else:
            assert current_readin == expected_readin
        env.close()
    with pytest.raises(ValueError, match="units"):
        load_reservoir(target, units=21)


def test_matrix_tampering_detected(parameter_row, tmp_path):
    """A modified W cannot masquerade as a selected matrix."""
    artifact = build_corrected(parameter_row, units=20)
    artifact["W"][0, 1] += 1
    save_reservoir(tmp_path / "bad.npz", artifact)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reservoir(tmp_path / "bad.npz")


def test_weighted_clustering_is_not_binary_clustering():
    """Weak triangle weights affect corrected Onnela clustering."""
    matrix = np.array([[0, 1, 0.01], [1, 0, 0.01], [0.01, 0.01, 0.0]])
    assignments = np.array([0, 0, 1])
    legacy = descriptors(matrix, assignments)
    corrected = descriptors(matrix, assignments, weighted=True)
    assert legacy["clustering"] == 1
    assert corrected["clustering"] < 0.1


@pytest.mark.parametrize("discrete", [False, True])
def test_corrected_action_and_reset(discrete):
    """Corrected input contains the actual action and resets state per episode."""
    env = wrapper(discrete=discrete)
    initial, _ = env.reset(seed=0)
    env.step(2 if discrete else np.array([0.25, -0.5]))
    encoded = env.last_input[2:-1]
    np.testing.assert_array_equal(encoded, [0, 0, 1] if discrete else [0.25, -0.5])
    repeated, _ = env.reset(seed=0)
    np.testing.assert_array_equal(initial, repeated)


def test_legacy_zero_action_preserved():
    """Existing checkpoint inputs retain the historical zero-action behavior."""
    env = wrapper("legacy")
    env.reset(seed=0)
    env.step(np.array([0.25, -0.5]))
    np.testing.assert_array_equal(env.last_input[2:-1], [0, 0])


def test_environment_factory_does_not_warm_legacy_state():
    """Initializing a legacy training environment feeds no extra observation."""
    from src.config import parse_args
    from src.training import make_environment, training_config

    args = parse_args("train_rl", ["--protocol", "legacy", "--env-id", "Swimmer-v4", "--smoke"])
    config = training_config(args)
    config["reservoir_seed"] = 42
    env = make_environment("Swimmer-v4", config)
    assert env.env.last_context is None
    assert all(np.count_nonzero(value) == 0 for value in env.env.reservoir.state.values())
    env.close()


def test_skip_connection_shape():
    """Skip connections append observation/action/reward to a one-dimensional state."""
    env = wrapper(skip=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (13,)
    obs, *_ = env.step(np.array([0.25, -0.5]))
    assert obs.shape == (13,)


def test_rollout_has_no_autoreset():
    """Recorded transitions belong to one episode and preserve terminal state."""
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(wrapper())
    result = rollout(FixedPolicy(), env, 0, collect=True)
    assert result["return"] == 3 and result["length"] == 3 and result["truncated"]
    assert result["states"].shape == (3, 8)
    assert env.env.env.counter == 3
    np.testing.assert_array_equal(result["states"][1:], result["post_states"][:-1])
    np.testing.assert_array_equal(result["terminal_state"], result["post_states"][-1])


@pytest.mark.integration
@pytest.mark.parametrize("task", TASKS)
def test_paper_velocity_masks(task):
    """Each paper locomotion task has a validated qvel mask and legal observation."""
    env = gym.make(task, max_episode_steps=5)
    old_shape = env.observation_space.shape[0]
    velocity_count = env.unwrapped.model.nv
    masked = DeletedVelocityWrapper(env)
    obs, _ = masked.reset(seed=0)
    assert len(obs) == old_shape - velocity_count
    assert masked.observation_space.contains(obs)
    masked.close()
