"""Shared argparse definitions and explicit legacy/corrected experiment presets."""

import argparse
import math

from src.utils import TASKS, get_random_seed, resolve_path

WORKFLOWS = {
    "select_reservoirs": "Select structurally diverse reservoirs without using RL outcomes.",
    "train_rl": "Train a single PPO configuration.",
    "train_all_selected": "Train all selected reservoirs and PPO baselines.",
    "train_and_plot": "Train the first reservoir per stratum with policy replicates.",
    "collect_obs_context_vec": "Evaluate checkpoints and collect episode-separated reservoir trajectories.",
    "evaluate": "Evaluate trained policies on seeded deterministic episodes.",
    "plot_context_embeddings": "Rank agents and plot reservoir activity with PCA and three CEBRA signals.",
    "inputdsa_100_reservoirs_H1": "Plot structural distance versus dynamical state distance (historical H1 filename).",
    "inputdsa_top_bottom_10_reservoirs_H1": "Plot structural/dynamical distances for top and bottom groups.",
    "inputdsa_100_reservoirs_H2": "Plot five InputDSA matrices (historical H2 filename; not paper H2).",
    "dev_scripts": "Analyze five InputDSA metrics for top/bottom models and test paper H1.",
    "inputdsa_5_systems_mackey_glass_ou": "Compare the first reservoir per stratum under synthetic inputs.",
    "inputdsa_100_reservoirs_mackey_glass_ou": "Compare all selected reservoirs under synthetic inputs.",
    "get_top_bottom_10_3_tasks": "Rank reservoirs using evaluation returns or historical TensorBoard rewards.",
    "descriptor_space_plot_performance": "Color structural PCA by reservoir performance.",
    "analyze_hypotheses": "Run Solution 1 H1, structure–dynamics inference, figures and separate paper H2.",
    "generate_fig_1": "Draw four canonical WSBM architectures.",
    "random_scores": "Evaluate random-action policies without training.",
}
TRAINING = {"train_rl", "train_all_selected", "train_and_plot"}
EVALUATION = {"evaluate", "collect_obs_context_vec", "plot_context_embeddings"}
SYNTHETIC = {"inputdsa_5_systems_mackey_glass_ou", "inputdsa_100_reservoirs_mackey_glass_ou"}
DSA_WORKFLOWS = SYNTHETIC | {
    "inputdsa_100_reservoirs_H1",
    "inputdsa_top_bottom_10_reservoirs_H1",
    "inputdsa_100_reservoirs_H2",
    "dev_scripts",
    "analyze_hypotheses",
}
PERFORMANCE = {
    "get_top_bottom_10_3_tasks",
    "descriptor_space_plot_performance",
    "dev_scripts",
    "inputdsa_top_bottom_10_reservoirs_H1",
    "analyze_hypotheses",
}


def positive(value):
    """Parse a strictly positive integer for counts and budgets."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def boolean(value):
    """Accept historical True/False arguments and conventional boolean spellings."""
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def finite_float(value):
    """Reject NaN and infinity in numerical experiment settings."""
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def add_bool(parser, name, default=None, help_text=""):
    """Add --flag [True|False], --no-flag, and historical underscore aliases."""
    parser.add_argument(
        *dict.fromkeys((f"--{name}", f"--{name.replace('-', '_')}")),
        dest=name.replace("-", "_"),
        nargs="?",
        const=True,
        type=boolean,
        default=default,
        help=help_text,
    )
    parser.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false", help=argparse.SUPPRESS)


def build_parser(workflow):
    """Build a workflow-specific parser without importing ML libraries."""
    parser = argparse.ArgumentParser(
        description=WORKFLOWS[workflow], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--protocol", choices=("legacy", "corrected"), required=True)
    parser.add_argument("--output-dir", type=str, help="Artifact root (explicit paths relative to caller)")
    parser.add_argument("--run-id", help="Unique name under artifact-root/protocol")
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        help="Explicitly permit reusing a run directory; never deletes files",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Validate and print configuration without writing or running",
    )
    parser.add_argument("--smoke", action="store_true", help="Execute a deliberately small validation experiment")
    parser.add_argument("--tasks", nargs="+", help="Gymnasium task IDs; corrected defaults to five paper tasks")
    parser.add_argument("--limit", type=positive, help="Use only the first N configurations, preserving csv_idx")
    parser.add_argument("--workers", type=positive, default=1, help="Parallel selection/DSA workers")
    parser.add_argument("--dpi", type=positive, default=150)
    defaults = {
        "selected_csv": None,
        "pool_csv": None,
        "models_csv": None,
        "evaluation_csv": None,
        "trajectories_csv": None,
        "models_dir": None,
        "logs_dir": None,
        "trajectories_dir": None,
        "units": None,
        "res_lr": None,
        "res_sr": 0.9,
        "res_iss": 1.0,
        "del_obs": None,
        "skip_c": False,
        "reset_res": None,
        "use_reservoir": True,
        "baseline": False,
        "motif": "assortative",
        "n_communities": 4,
        "hi": 0.9,
        "lo": 0.1,
        "mid": 0.5,
        "sigma": 0.1,
        "connectivity": 0.1,
        "symmetric": True,
        "p_negative": 0.0,
        "training_steps": 500000,
        "learning_rate": 3e-4,
        "n_envs": 1,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "reservoir_seed": None,
        "environment_seed": 0,
        "policy_seeds": None,
        "seed": None,
        "evaluation_seed": 10000,
        "evaluation_episodes": 10,
        "max_episode_steps": None,
        "device": "cpu",
        "backend": None,
        "n_delays": 3,
        "rank": None,
        "rank_energy": 0.99,
        "min_rank": 1,
        "max_rank": 50,
        "analysis_seed": 0,
        "dmd_regularization": 1e-8,
        "state_iters": 1500,
        "state_learning_rate": 1e-3,
        "state_metric": "wasserstein",
        "allow_subset": False,
        "group_size": 10,
        "ranking_source": None,
        "weight_type": "linear",
        "window_steps": 10000,
        "permutations": 9999,
        "input_steps": 3000,
        "input_dim": 10,
        "input_seed": 0,
        "ou_theta": 0.15,
        "conditions": ["mackey_glass", "ou_noise"],
        "pool_per_stratum": 1000,
        "select_per_stratum": 20,
        "selection_seed": 0,
        "pca_components": 5,
        "sigma_base": 0.1,
        "strata": None,
        "figure_seed": 42,
        "communities": [2, 3, 4, 6, 8, 12, 16],
        "density_range": [0.03, 0.30],
        "centre_range": [0, 10],
        "dispersion_range": [0.125, 8],
        "topology_range": [0.0625, 16],
        "size_alphas": [float("inf"), 100, 30, 10, 3],
        "contrast_range": [0.5, 8],
        "core_range": [0.05, 0.40],
        "mixing_range": [0.2, 0.8],
        "null_centre_range": [2, 10],
        "core_contrast_range": [1, 8],
        "mixed_contrast_range": [1, 8],
        "figure_hi": 0.9,
        "figure_lo": 0.1,
        "figure_mid": 0.5,
        "figure_connectivity": 0.1,
        "figure_core_fraction": 0.2,
        "figure_layout": "community",
    }
    parser.set_defaults(**defaults)
    if workflow in TRAINING:
        parser.add_argument("--resume", action="store_true", help="Resume an existing run, skipping completed policies")
        parser.add_argument(
            "--checkpoint-steps",
            type=positive,
            default=10000,
            help="Save progress after PPO updates at approximately this many environment timesteps",
        )
    if workflow not in {"select_reservoirs", "generate_fig_1", "random_scores"}:
        parser.add_argument(
            "--selected-csv",
            "--selected_csv",
            help="Explicit selection; legacy default is repository root selected.csv",
        )
    if workflow in {"descriptor_space_plot_performance", "analyze_hypotheses"} | DSA_WORKFLOWS:
        parser.add_argument("--pool-csv", help="Full structural pool; corrected uses sibling descriptors.csv")
    if workflow in TRAINING | EVALUATION | SYNTHETIC | {"select_reservoirs", "generate_fig_1"}:
        parser.add_argument("--units", type=positive, help="Neurons per reservoir, distinct from configuration count")
        parser.add_argument("--res-lr", "--res_lr", type=finite_float, help="Reservoir leak rate")
        parser.add_argument("--res-sr", "--res_sr", type=finite_float, default=0.9)
        parser.add_argument("--res-iss", "--res_iss", type=finite_float, default=1.0)
    if workflow in TRAINING | EVALUATION:
        parser.add_argument("--env-id", "--env_id", help="Single environment; overrides default tasks")
        parser.add_argument(
            "--training-steps", "--training_steps", type=positive, default=300000 if workflow == "train_rl" else 500000
        )
        parser.add_argument("--policy-seeds", nargs="+", type=int, help="Independent policy optimization seeds")
        parser.add_argument(
            "--seeds", type=positive, help="Compatibility alias: number of policy seeds starting at zero"
        )
        parser.add_argument("--seed", type=int, help="Legacy single training seed; -1 uses SLURM or zero")
        parser.add_argument(
            "--reservoir-seed", type=int, help="Legacy structure seed; corrected uses selection metadata"
        )
        parser.add_argument("--environment-seed", type=int, default=0)
        parser.add_argument("--learning-rate", "--learning_rate", type=finite_float, default=3e-4)
        parser.add_argument("--n-envs", "--n_envs", type=positive, default=1)
        parser.add_argument("--n-steps", type=positive, default=2048)
        parser.add_argument("--batch-size", type=positive, default=64)
        parser.add_argument("--n-epochs", type=positive, default=10)
        for name, value in (
            ("gamma", 0.99),
            ("gae-lambda", 0.95),
            ("clip-range", 0.2),
            ("ent-coef", 0),
            ("vf-coef", 0.5),
            ("max-grad-norm", 0.5),
        ):
            parser.add_argument(f"--{name}", type=finite_float, default=value)
        add_bool(parser, "del-obs", help_text="Mask velocity observations (corrected default True)")
        add_bool(parser, "reset-res", help_text="Reset reservoir at each episode (corrected default True)")
        add_bool(parser, "skip-c", False, "Concatenate raw input to context")
        add_bool(parser, "use-reservoir", True, "Train reservoir policies")
        add_bool(parser, "baseline", workflow != "train_rl", "Include vanilla PPO with the same observation mask")
        parser.add_argument(
            "--motif",
            choices=("assortative", "disassortative", "core_periphery", "mixed", "null"),
            default="assortative",
        )
        parser.add_argument("--n-communities", "--n_communities", type=positive, default=4)
        parser.add_argument("--csv-idx", "--csv_idx", type=int, help="Select one original CSV row by stable identifier")
        for name, value in (
            ("hi", 0.9),
            ("lo", 0.1),
            ("mid", 0.5),
            ("sigma", 0.1),
            ("connectivity", 0.1),
            ("p-negative", 0),
        ):
            parser.add_argument(
                *dict.fromkeys((f"--{name}", f"--{name.replace('-', '_')}")), type=finite_float, default=value
            )
        add_bool(parser, "symmetric", True)
    if workflow in TRAINING | EVALUATION | DSA_WORKFLOWS | {"random_scores"}:
        parser.add_argument("--device", default="cpu", help="Torch training/DSA device")
    if workflow in EVALUATION:
        parser.add_argument("--models-csv", help="Model registry generated by training")
        parser.add_argument("--models-dir", "--models_dir", help="Read-only legacy checkpoint root")
        parser.add_argument("--logs-dir", "--log_dir", help="Read-only legacy TensorBoard root")
    if workflow in EVALUATION | {"random_scores"}:
        parser.add_argument("--evaluation-episodes", type=positive, default=100 if workflow == "random_scores" else 10)
        parser.add_argument("--evaluation-seed", type=int, default=10000)
    if workflow in TRAINING | EVALUATION | {"random_scores"}:
        parser.add_argument("--max-episode-steps", type=positive, help="Explicit Gym time limit; saved in model config")
    if workflow == "plot_context_embeddings":
        parser.add_argument("--evaluation-csv", help="Saved deterministic per-episode returns with run manifest")
        parser.add_argument("--trajectories-csv", help="Saved episode trajectory registry")
        parser.add_argument("--group", choices=("top", "bottom", "both"), default="top")
        parser.add_argument("--k", type=positive, default=1)
        parser.add_argument("--embedding-mode", choices=("separate", "shared"), default="separate")
        parser.add_argument("--analysis-seed", type=int, default=0)
        parser.add_argument("--cebra-iterations", type=positive, default=10000)
        parser.add_argument("--cebra-batch-size", type=positive, default=512)
        parser.add_argument("--cebra-learning-rate", type=finite_float, default=3e-4)
        parser.add_argument("--cebra-delta", type=finite_float, default=0.1)
    if workflow in DSA_WORKFLOWS:
        parser.add_argument("--backend", choices=("dmdc", "n4sid"))
        parser.add_argument("--n-delays", type=positive, default=10 if workflow in SYNTHETIC else 3)
        parser.add_argument("--rank", type=positive, help="Explicit common rank; default uses energy selection")
        parser.add_argument("--rank-energy", type=finite_float, default=0.99)
        parser.add_argument(
            "--min-rank", type=positive, default=3 if workflow == "inputdsa_100_reservoirs_mackey_glass_ou" else 1
        )
        parser.add_argument("--max-rank", type=positive, default=50)
        parser.add_argument("--dmd-regularization", type=finite_float, default=1e-8)
        parser.add_argument("--state-iters", type=positive, default=1500)
        parser.add_argument("--state-learning-rate", type=finite_float, default=1e-3)
        parser.add_argument("--state-metric", choices=("wasserstein", "euclidean", "angular"), default="wasserstein")
        parser.add_argument("--analysis-seed", type=int, default=0)
        parser.add_argument(
            "--allow-subset", action="store_true", help="Explicitly permit missing systems and export exclusions"
        )
        if workflow not in SYNTHETIC:
            parser.add_argument(
                "--policy-seeds",
                nargs="+",
                type=int,
                help="Required policy replicates for corrected dynamics (defaults to 0–4)",
            )
            parser.add_argument(
                "--dynamics-episodes",
                "--evaluation-episodes",
                dest="evaluation_episodes",
                type=positive,
                default=10,
                help="Required episodes per policy dynamics fit",
            )
            parser.add_argument("--trajectories-csv", help="Episode registry generated by collection")
            parser.add_argument("--trajectories-dir", help="Read-only legacy Ot/Ct .npy directory")
    if workflow in PERFORMANCE:
        parser.add_argument("--evaluation-csv", help="Deterministic episode returns, with original run manifest")
        parser.add_argument("--ranking-source", choices=("evaluation", "tensorboard"))
        parser.add_argument("--group-size", type=positive, default=10)
        parser.add_argument("--window-steps", type=positive, default=10000)
        parser.add_argument("--weight-type", choices=("linear", "exponential", "mean"), default="linear")
        parser.add_argument("--logs-dir", help="Read-only historical log root")
        parser.add_argument("--models-dir", help="Read-only historical model root")
        parser.add_argument("--training-steps", type=positive, default=500000)
    if workflow in {"analyze_hypotheses", "dev_scripts"}:
        parser.add_argument("--permutations", type=positive, default=9999)
    if workflow == "analyze_hypotheses":
        parser.add_argument("--interim", action="store_true", help="Analyze complete tasks before all five finish")
        parser.add_argument("--bootstrap-replicates", type=positive, default=5000)
        parser.add_argument("--models-csv", help="Training registry for settings, checkpoint and learning-curve audit")
        parser.add_argument("--distance-cache", type=str, help="Reuse a matching Solution 1 run's fitted distances")
        parser.add_argument(
            "--exploratory-reuse-ranking-episodes",
            action="store_true",
            help="Explicitly allow old ranking episodes for exploratory dynamics; never confirmatory",
        )
    if workflow in SYNTHETIC:
        parser.add_argument("--input-steps", type=positive, default=2000 if "5_systems" in workflow else 3000)
        parser.add_argument("--input-dim", type=positive, default=10)
        parser.add_argument("--input-seed", type=int, default=0)
        parser.add_argument("--ou-theta", type=finite_float, default=0.15)
        parser.add_argument(
            "--conditions", nargs="+", choices=("mackey_glass", "ou_noise"), default=["mackey_glass", "ou_noise"]
        )
    if workflow == "select_reservoirs":
        parser.add_argument("pool_count", nargs="?", type=positive, help="Historical positional pool size")
        parser.add_argument("--pool-per-stratum", type=positive, default=1000)
        parser.add_argument("--select-per-stratum", type=positive, default=20)
        parser.add_argument("--selection-seed", type=int, default=0)
        parser.add_argument("--pca-components", type=positive, default=5)
        parser.add_argument("--sigma-base", type=finite_float, default=0.1)
        parser.add_argument(
            "--strata", nargs="+", choices=("assortative", "disassortative", "core_periphery", "mixed", "null")
        )
        parser.add_argument("--communities", nargs="+", type=positive, default=[2, 3, 4, 6, 8, 12, 16])
        parser.add_argument("--size-alphas", nargs="+", type=float, default=[float("inf"), 100, 30, 10, 3])
        for name, value in (
            ("density-range", (0.03, 0.30)),
            ("centre-range", (0, 10)),
            ("dispersion-range", (0.125, 8)),
            ("topology-range", (0.0625, 16)),
            ("contrast-range", (0.5, 8)),
            ("core-range", (0.05, 0.4)),
            ("mixing-range", (0.2, 0.8)),
            ("null-centre-range", (2, 10)),
            ("core-contrast-range", (1, 8)),
            ("mixed-contrast-range", (1, 8)),
        ):
            parser.add_argument(f"--{name}", nargs=2, type=finite_float, default=value, metavar=("MIN", "MAX"))
    if workflow == "generate_fig_1":
        parser.add_argument("--n-communities", type=positive, default=6)
        parser.add_argument("--figure-seed", type=int, default=42)
        parser.add_argument("--sigma-base", type=finite_float, default=0.1)
        for flag, default in (("hi", 0.9), ("lo", 0.1), ("mid", 0.5), ("connectivity", 0.1), ("core-fraction", 0.2)):
            parser.add_argument(
                f"--{flag}", dest="figure_" + flag.replace("-", "_"), type=finite_float, default=default
            )
        parser.add_argument("--layout", dest="figure_layout", choices=("community", "spring"), default="community")
    if workflow in {"dev_scripts", "descriptor_space_plot_performance"}:
        parser.add_argument("task_names", nargs="*", help="Historical positional task IDs")
    return parser


def parse_args(workflow, argv=None):
    """Resolve explicit presets, validate numeric settings, and apply smoke budgets."""
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(workflow)
    args = parser.parse_args(arguments)
    if getattr(args, "resume", False) and (not args.run_id or args.overwrite):
        parser.error("--resume requires --run-id and cannot be combined with --overwrite")
    corrected = args.protocol == "corrected"
    args.requested_units = args.units
    args.units = args.units or (
        200 if corrected or workflow == "select_reservoirs" else 68 if workflow == "generate_fig_1" else 100
    )
    args.res_lr = (
        args.res_lr
        if args.res_lr is not None
        else 0.3
        if workflow in SYNTHETIC | {"generate_fig_1", "select_reservoirs"} and not corrected
        else 0.1
    )
    args.del_obs = corrected if args.del_obs is None else args.del_obs
    args.reset_res = corrected if args.reset_res is None else args.reset_res
    args.backend = args.backend or (
        "dmdc"
        if corrected or workflow in {"inputdsa_100_reservoirs_H1", "inputdsa_top_bottom_10_reservoirs_H1"}
        else "n4sid"
    )
    args.ranking_source = args.ranking_source or ("evaluation" if corrected else "tensorboard")
    if (
        corrected
        and workflow == "analyze_hypotheses"
        and not any(item.startswith("--group-size") for item in arguments)
    ):
        args.group_size = 5
    legacy_tasks = {
        "random_scores": [
            "Ant-v4",
            "HalfCheetah-v4",
            "Hopper-v4",
            "Humanoid-v4",
            "HumanoidStandup-v4",
            "InvertedDoublePendulum-v4",
            "InvertedPendulum-v4",
            "Pusher-v4",
            "Reacher-v4",
            "Swimmer-v4",
            "Walker2d-v4",
        ],
        "inputdsa_100_reservoirs_H1": ["Humanoid-v4"],
        "inputdsa_top_bottom_10_reservoirs_H1": ["Walker2d-v4"],
        "get_top_bottom_10_3_tasks": ["Swimmer-v4", "Hopper-v4", "Walker2d-v4"],
    }
    args.tasks = (
        args.tasks or list(TASKS)
        if corrected
        else args.tasks or legacy_tasks.get(workflow, ["Ant-v4", "HalfCheetah-v4", "Swimmer-v4"])
    )
    if (
        not corrected
        and workflow == "dev_scripts"
        and not getattr(args, "task_names", None)
        and not any(item.startswith("--tasks") for item in arguments)
    ):
        args.tasks = ["HalfCheetah-v4", "Ant-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4"]
    if getattr(args, "task_names", None):
        args.tasks = args.task_names
    if getattr(args, "env_id", None):
        args.tasks = [args.env_id]
    if (
        workflow == "plot_context_embeddings"
        and not any(t.startswith("--tasks") for t in arguments)
        and not args.env_id
    ):
        args.tasks = list(TASKS)
    if (
        workflow == "train_rl"
        and not getattr(args, "env_id", None)
        and not any(item.startswith("--tasks") for item in arguments)
    ):
        parser.error("train_rl requires --env-id or explicit --tasks")
    if args.seed is not None:
        args.policy_seeds = [get_random_seed() if args.seed == -1 else args.seed]
    elif getattr(args, "seeds", None):
        args.policy_seeds = list(range(args.seeds))
    args.policy_seeds = args.policy_seeds or (
        [get_random_seed()]
        if workflow == "train_rl" and not corrected
        else list(range(5 if corrected or workflow == "train_and_plot" else 1))
    )
    if len(set(args.policy_seeds)) != len(args.policy_seeds) or any(seed < 0 for seed in args.policy_seeds):
        parser.error("policy seeds must be unique and nonnegative")
    if corrected and args.reservoir_seed is not None:
        parser.error("corrected structure seeds come from selected matrices; omit --reservoir-seed")
    structural_flags = {
        "--motif",
        "--n-communities",
        "--n_communities",
        "--hi",
        "--lo",
        "--mid",
        "--sigma",
        "--connectivity",
        "--p-negative",
        "--p_negative",
        "--symmetric",
        "--no-symmetric",
    }
    if (
        corrected
        and workflow in TRAINING | EVALUATION
        and any(token.split("=", 1)[0] in structural_flags for token in arguments)
    ):
        parser.error("corrected structural settings come from selected matrices; change the selection instead")
    args.selected_csv = (
        resolve_path(args.selected_csv, "selected.csv") if args.selected_csv is not None or not corrected else None
    )
    for name, default in (
        ("models_dir", "rl_only/models_all"),
        ("logs_dir", "rl_only/logs_all"),
        ("trajectories_dir", "trajectories"),
    ):
        setattr(args, name, resolve_path(getattr(args, name), default))
    for name in ("pool_csv", "models_csv", "evaluation_csv", "trajectories_csv"):
        if getattr(args, name) is not None:
            setattr(args, name, resolve_path(getattr(args, name), ""))
    if workflow == "select_reservoirs" and args.pool_count:
        args.pool_per_stratum = args.pool_count
    if args.smoke:
        args.limit, args.policy_seeds, args.evaluation_episodes = args.limit or 4, [0], 1
        if workflow != "plot_context_embeddings":
            args.training_steps = 64
        args.n_steps, args.batch_size, args.n_epochs = 32, 16, 1
        args.max_episode_steps = args.max_episode_steps or 100
        args.tasks = [args.tasks[0]]
        args.permutations, args.state_iters, args.max_rank = 19, 10, min(args.max_rank, 3)
        args.group_size, args.input_steps, args.n_delays = 2, 100, 1
        args.min_rank = min(args.min_rank, args.max_rank)
        if workflow == "select_reservoirs":
            args.units, args.communities = 20, [2, 3]
            args.pool_per_stratum, args.select_per_stratum = 5, 1
            args.density_range, args.topology_range = [0.15, 0.25], [0.5, 2]
        if workflow == "plot_context_embeddings":
            args.cebra_iterations, args.cebra_batch_size = 10, 32
        if workflow == "analyze_hypotheses":
            args.bootstrap_replicates = 50
    if not 0 < args.res_lr <= 1 or not 0 < args.res_sr < 1 or args.res_iss <= 0:
        parser.error("require 0 < leak rate <= 1, 0 < spectral radius < 1, and positive input scaling")
    if workflow == "plot_context_embeddings" and (args.cebra_learning_rate <= 0 or args.cebra_delta <= 0):
        parser.error("CEBRA learning rate and delta must be positive")
    if not 0 < args.rank_energy <= 1 or args.min_rank > args.max_rank or args.dmd_regularization < 0:
        parser.error("invalid rank energy/bounds or negative DMD regularization")
    if args.learning_rate <= 0 or args.batch_size > args.n_steps * args.n_envs or args.n_steps * args.n_envs < 2:
        parser.error("invalid PPO learning rate or rollout/batch sizes")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1 or args.clip_range <= 0 or args.max_grad_norm <= 0:
        parser.error("invalid PPO discount, clipping, or gradient parameters")
    if any(
        seed < 0
        for seed in (
            args.environment_seed,
            args.evaluation_seed,
            args.analysis_seed,
            args.selection_seed,
            args.input_seed,
        )
    ):
        parser.error("seeds must be nonnegative")
    if workflow == "select_reservoirs":
        if args.select_per_stratum > args.pool_per_stratum or args.pca_components > 8 or args.sigma_base <= 0:
            parser.error("invalid selection count, PCA dimensions, or weight dispersion")
        if any(k > args.units for k in args.communities) or any(a <= 0 or math.isnan(a) for a in args.size_alphas):
            parser.error("community counts must fit units and size alphas must be positive")
        for name in (
            "density_range",
            "centre_range",
            "dispersion_range",
            "topology_range",
            "contrast_range",
            "core_range",
            "mixing_range",
            "null_centre_range",
            "core_contrast_range",
            "mixed_contrast_range",
        ):
            lower, upper = getattr(args, name)
            if lower > upper or (name not in {"centre_range", "null_centre_range"} and lower <= 0):
                parser.error(f"invalid {name}")
        if args.density_range[1] > 1 or args.core_range[1] >= 1 or args.mixing_range[1] > 1:
            parser.error("density/core/mixing ranges exceed probability bounds")
    if args.smoke and workflow in DSA_WORKFLOWS:
        args.min_rank = min(args.min_rank, args.max_rank)
    if workflow == "generate_fig_1":
        if (
            args.n_communities < 2
            or args.n_communities > args.units
            or args.sigma_base <= 0
            or not 0 < args.figure_core_fraction < 1
            or not 0 <= args.figure_connectivity <= 1
            or not args.figure_lo < args.figure_mid < args.figure_hi
        ):
            parser.error("invalid figure community, block-mean, dispersion, or probability settings")
    return args
