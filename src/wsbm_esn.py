"""Standalone reference WSBM ESNs and a ridge readout.

The RL pipeline uses the ReservoirPy fork instead. This reference module remains
available for structural figures and supervised sequence experiments::

    from src.wsbm_esn import AssortativeESN
    esn = AssortativeESN(n_reservoir=20, random_state=42)
    matrix, assignments = esn.W_res, esn.community_assignments
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

# Import utility functions.
# There is no utils.py in this project, so this import failed and made the
# whole module unimportable -- including the models that need no connectome
# data at all. Degrade instead: the loaders return None and every
# Connectome*ESN already handles that via its documented random-matrix
# fallback. Drop a real utils.py next to this file and it takes over.
try:
    from utils import (
        load_connectivity_matrices,
        load_functional_connectome,
        load_structural_connectome,
    )
except ImportError:  # pragma: no cover

    def load_structural_connectome(*args, **kwargs):
        """Return None when optional connectome loaders are unavailable."""
        return None

    def load_functional_connectome(*args, **kwargs):
        """Return None when optional connectome loaders are unavailable."""
        return None

    def load_connectivity_matrices(*args, **kwargs):
        """Return an empty mapping when optional connectome loaders are unavailable."""
        return {}


#######################
# Base ESN Class
#######################
class BaseESN(ABC):
    """
    Base class for all Echo State Network implementations.

    This abstract class defines the common interface and functionality
    for all ESN variants. Specific implementations should inherit from this
    class and implement the required methods.
    """

    def __init__(
        self,
        n_reservoir: int = 68,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        ridge_regression_alpha: float = 1e-6,
        random_state: Optional[int] = None,
        input_dim: int = 1,
    ):
        """
        Initialize the base ESN parameters.

        Args:
            n_reservoir: Number of reservoir units
            input_scaling: Scaling factor for input weights
            spectral_radius: Desired spectral radius of the reservoir
            leak_rate: Leak rate for the reservoir units (0-1)
            ridge_regression_alpha: Regularization parameter for ridge regression
            random_state: Random seed for reproducibility
            input_dim: Dimensionality of the input data
        """
        self.n_reservoir = n_reservoir
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.ridge_regression_alpha = ridge_regression_alpha
        self.input_dim = input_dim

        # Keep the seed: subclasses (e.g. KMeansCESN) pass it on to sklearn.
        self.random_state = random_state
        # Per-instance generator. Prefer this over np.random.* in new code --
        # the global seeding below makes results depend on construction order.
        self.rng = np.random.default_rng(random_state)

        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize weight matrices
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_out = None  # Output weights
        self.readout = None  # Readout model

        # State tracking
        self.state_history = None

    @abstractmethod
    def initialize_weights(self):
        """
        Initialize the weight matrices.
        Each ESN variant should implement its own weight initialization.
        """
        pass

    def _update_state(self, state: np.ndarray, input_pattern: np.ndarray) -> np.ndarray:
        """
        Update the reservoir state for a single time step.

        Standard leaky-integrator ESN update:

            x(t+1) = (1 - a) x(t) + a * tanh(W_in u(t) + W_res x(t))

        This used to be abstract, which meant ten subclasses each carried a
        byte-identical copy of it. Override only if your variant genuinely
        differs (DendriticESN and MultiCycleReservoir do).

        Args:
            state: Current reservoir state
            input_pattern: Input at the current time step

        Returns:
            Updated reservoir state
        """
        input_contribution = np.dot(self.W_in, np.reshape(input_pattern, (-1, 1))).flatten()
        reservoir_contribution = np.dot(self.W_res, state)

        return (1 - self.leak_rate) * state + self.leak_rate * np.tanh(input_contribution + reservoir_contribution)

    def train(self, X: np.ndarray, y: np.ndarray, track_states: bool = False):
        """
        Train the ESN using ridge regression.

        Args:
            X: Input time series (n_samples, input_dim)
            y: Target time series (n_samples, output_dim)
            track_states: Whether to store state history during training
        """
        # Ensure X has correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Validate input dimension
        if self.input_dim != X.shape[1]:
            self.input_dim = X.shape[1]
            self.initialize_weights()

        # Initialize reservoir states
        states = np.zeros((len(X), self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        # Run reservoir for input sequence
        for t in range(len(X)):
            state = self._update_state(state, X[t])
            states[t] = state

        # Store state history if tracking is enabled
        if track_states:
            self.state_history = states.copy()

        # Train readout using ridge regression
        self.readout = Ridge(alpha=self.ridge_regression_alpha)
        self.readout.fit(states, y)

        # Store output weights for direct access
        self.W_out = self.readout.coef_

        return self

    def predict(self, X: np.ndarray, track_states: bool = False) -> np.ndarray:
        """
        Generate predictions using the trained ESN.

        Args:
            X: Input time series (n_samples, input_dim)
            track_states: Whether to store state history during prediction

        Returns:
            Predicted time series (n_samples, output_dim)
        """
        # Ensure X has correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Initialize state and predictions
        state = np.zeros(self.n_reservoir)
        predictions = []
        states = np.zeros((len(X), self.n_reservoir)) if track_states else None

        # Run reservoir for input sequence
        for t in range(len(X)):
            state = self._update_state(state, X[t])
            if track_states:
                states[t] = state
            predictions.append(self.readout.predict(state.reshape(1, -1))[0])

        # Store state history if tracking is enabled
        if track_states:
            self.state_history = states

        return np.array(predictions)

    def get_state_data(self) -> Optional[np.ndarray]:
        """
        Get the recorded state history if available.

        Returns:
            State history if tracked, None otherwise
        """
        return self.state_history

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and format input data.

        Args:
            X: Input data

        Returns:
            Properly formatted input data
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {X.shape[1]}")

        return X

    def reset_state(self):
        """Reset the internal state of the reservoir."""
        self.state_history = None

    def visualize_structure(self, save_path=None, interactive=False):
        """
        Visualize the network structure.

        Parameters:
        -----------
        save_path : str
            Path to save the visualization. If None, the plot is displayed.
        interactive : bool
            Whether to create an interactive visualization (requires plotly).
        """
        if interactive:
            try:
                import networkx as nx
                import plotly.graph_objects as go

                # Create a directed graph
                G = nx.DiGraph()

                # Add reservoir neurons
                for i in range(self.n_reservoir):
                    G.add_node(f"res_{i}", type="reservoir")

                # Add input nodes
                for i in range(self.input_dim):
                    G.add_node(f"in_{i}", type="input")

                # Add input connections
                for i in range(self.n_reservoir):
                    for j in range(self.input_dim):
                        if self.W_in is not None and np.abs(self.W_in[i, j]) > 0.01:  # threshold
                            G.add_edge(f"in_{j}", f"res_{i}", weight=self.W_in[i, j])

                # Add reservoir connections
                for i in range(self.n_reservoir):
                    for j in range(self.n_reservoir):
                        if self.W_res is not None and np.abs(self.W_res[i, j]) > 0.01:  # threshold
                            G.add_edge(f"res_{i}", f"res_{j}", weight=self.W_res[i, j])

                # Create positions
                pos = nx.spring_layout(G, seed=42)

                # Create separate traces for input-to-reservoir and reservoir-to-reservoir edges
                input_edges = [(u, v) for u, v in G.edges() if "in_" in u]
                res_edges = [(u, v) for u, v in G.edges() if "res_" in u]

                # Input to reservoir edge trace
                input_edge_x = []
                input_edge_y = []
                for edge in input_edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    input_edge_x.extend([x0, x1, None])
                    input_edge_y.extend([y0, y1, None])

                input_edge_trace = go.Scatter(
                    x=input_edge_x,
                    y=input_edge_y,
                    line=dict(width=0.5, color="rgba(50, 150, 50, 0.5)"),
                    hoverinfo="none",
                    mode="lines",
                )

                # Reservoir to reservoir edge trace
                res_edge_x = []
                res_edge_y = []
                for edge in res_edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    res_edge_x.extend([x0, x1, None])
                    res_edge_y.extend([y0, y1, None])

                res_edge_trace = go.Scatter(
                    x=res_edge_x,
                    y=res_edge_y,
                    line=dict(width=0.5, color="rgba(150, 50, 50, 0.5)"),
                    hoverinfo="none",
                    mode="lines",
                )

                # Create node traces
                input_nodes_x = []
                input_nodes_y = []
                res_nodes_x = []
                res_nodes_y = []

                for node in G.nodes():
                    x, y = pos[node]
                    if "in_" in node:
                        input_nodes_x.append(x)
                        input_nodes_y.append(y)
                    else:
                        res_nodes_x.append(x)
                        res_nodes_y.append(y)

                input_node_trace = go.Scatter(
                    x=input_nodes_x,
                    y=input_nodes_y,
                    mode="markers",
                    marker=dict(size=8, color="blue"),
                    hoverinfo="text",
                    text=[f"Input {i}" for i in range(len(input_nodes_x))],
                )

                res_node_trace = go.Scatter(
                    x=res_nodes_x,
                    y=res_nodes_y,
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    hoverinfo="text",
                    text=[f"Reservoir {i}" for i in range(len(res_nodes_x))],
                )

                # Create figure
                data = [input_edge_trace, res_edge_trace, input_node_trace, res_node_trace]
                fig = go.Figure(
                    data=data,
                    layout=go.Layout(
                        title="ESN Network Structure",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ),
                )

                if save_path:
                    fig.write_html(save_path)
                else:
                    fig.show()

                return G

            except ImportError as e:
                print(f"Interactive visualization requires plotly and networkx: {e}")
                print("Falling back to static visualization")
                interactive = False

        # Static visualization
        if not interactive:
            plt.figure(figsize=(10, 8))

            # Create adjacency matrix for visualization
            adjacency = np.zeros((self.input_dim + self.n_reservoir, self.input_dim + self.n_reservoir))

            # Input to reservoir connections
            if self.W_in is not None:
                for i in range(self.n_reservoir):
                    for j in range(self.input_dim):
                        adjacency[j, self.input_dim + i] = np.abs(self.W_in[i, j])

            # Reservoir connections
            if self.W_res is not None:
                for i in range(self.n_reservoir):
                    for j in range(self.n_reservoir):
                        adjacency[self.input_dim + i, self.input_dim + j] = np.abs(self.W_res[i, j])

            plt.imshow(adjacency, cmap="viridis")
            plt.colorbar(label="Connection Strength")

            # Add labels
            plt.axhline(y=self.input_dim - 0.5, color="white", linestyle="-", alpha=0.3)
            plt.axvline(x=self.input_dim - 0.5, color="white", linestyle="-", alpha=0.3)

            plt.title("ESN Connectivity")
            plt.xlabel("To")
            plt.ylabel("From")

            labels = [f"In {i}" for i in range(self.input_dim)] + [f"Res {i}" for i in range(self.n_reservoir)]
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.yticks(range(len(labels)), labels)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

    def visualize_activations(self, X, save_path=None, interactive=False):
        """
        Visualize reservoir activations for a given input sequence.

        Parameters:
        -----------
        X : array
            Input data with shape (n_samples, input_dim)
        save_path : str
            Path to save the visualization. If None, the plot is displayed.
        interactive : bool
            Whether to create an interactive heatmap (requires plotly).
        """
        # Ensure X has correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get activations
        state = np.zeros(self.n_reservoir)
        states = []

        # Process at most 100 time steps to keep visualization manageable
        for t in range(min(100, len(X))):
            state = self._update_state(state, X[t])
            states.append(np.abs(state))

        states = np.array(states)

        if interactive:
            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    data=go.Heatmap(
                        z=states,
                        x=[f"Neuron {i}" for i in range(self.n_reservoir)],
                        y=[f"Time {t}" for t in range(states.shape[0])],
                        colorscale="Viridis",
                    )
                )

                fig.update_layout(
                    title="Reservoir Neuron Activations", xaxis_title="Reservoir Neurons", yaxis_title="Time Steps"
                )

                if save_path:
                    fig.write_html(save_path)
                else:
                    fig.show()

            except ImportError as e:
                print(f"Interactive visualization requires plotly: {e}")
                print("Falling back to static visualization")
                interactive = False

        if not interactive:
            plt.figure(figsize=(14, 8))

            # Heatmap of activations
            plt.subplot(2, 1, 1)
            plt.imshow(states, aspect="auto", cmap="viridis")
            plt.colorbar(label="Activation")
            plt.xlabel("Reservoir Neurons")
            plt.ylabel("Time Steps")
            plt.title("Reservoir Neuron Activations")

            # Line plot of selected neurons
            plt.subplot(2, 1, 2)
            selected_neurons = min(5, self.n_reservoir)
            for i in range(selected_neurons):
                plt.plot(states[:, i], label=f"Neuron {i}")

            plt.xlabel("Time Steps")
            plt.ylabel("Activation")
            plt.title(f"Activation of {selected_neurons} Selected Neurons")
            plt.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()


#######################
# WSBM-based ESN Models
#######################
#
# One generative equation (Betzel, Medaglia & Bassett 2018, Nat Commun 9:346):
#
#     W[i,j] ~ N( omega[z_i, z_j], sigma^2 )  masked by  Bernoulli(connectivity)
#
# `z` is the node -> community assignment and `omega` is the KxK matrix of
# block mean weights. The four "architectures" are not four models -- they are
# four choices of `omega`. The paper classifies a *pair* of communities {r,s}
# purely by the ordering of omega_rr, omega_ss, omega_rs (Fig. 6b / Methods):
#
#     assortative      if min(w_rr, w_ss) > w_rs
#     core-periphery   if w_rr > w_rs > w_ss   (or the mirror image)
#     disassortative   if w_rs > max(w_rr, w_ss)
#
# "Mixed" is not a fifth rule: it is an omega whose pairs fall into several of
# the classes above.
#
# This replaces four hand-written classes that were deleted for being broken:
# Assortative and Disassortative had byte-identical initialize_weights (so the
# disassortative model built assortative networks), neither defined
# _update_state so both were uninstantiable, and none was actually a block
# model -- edge (i,j) was indexed by i's community alone, so omega depended on
# r but not on the pair {r,s}.

MOTIFS = ("assortative", "disassortative", "core_periphery", "mixed")


def classify_motif(w_rr: float, w_ss: float, w_rs: float, tol: float = 1e-12) -> str:
    """Classify one community pair. Betzel et al. 2018, Methods eq. (9)."""
    if abs(w_rr - w_rs) <= tol and abs(w_ss - w_rs) <= tol:
        return "degenerate"  # all three equal -> the rule does not discriminate
    if min(w_rr, w_ss) > w_rs:
        return "assortative"
    if w_rs > max(w_rr, w_ss):
        return "disassortative"
    return "core_periphery"


def classify_omega(omega: np.ndarray) -> Dict[Tuple[int, int], str]:
    """Motif class of every community pair r < s."""
    K = len(omega)
    return {(r, s): classify_motif(omega[r, r], omega[s, s], omega[r, s]) for r in range(K) for s in range(r + 1, K)}


def build_omega(
    motif: str,
    n_communities: int,
    hi: float = 0.9,
    lo: float = 0.1,
    mid: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """The KxK block-mean matrix for each canonical architecture.

    ===================================================================
    WHICH PARAMETERS MAKE WHICH ARCHITECTURE
    ===================================================================
    Only `omega` changes between the four types. Everything else about the
    model -- sampling, density, spectral rescaling, the state update -- is
    identical. With hi=0.9, lo=0.1, mid=0.5 and K=3, omega looks like:

      motif="assortative"        strong on the diagonal
          diagonal (w_rr) = hi        [[.9 .1 .1]
          off-diag (w_rs) = lo         [.1 .9 .1]
                                       [.1 .1 .9]]
          -> min(w_rr, w_ss) > w_rs

      motif="disassortative"     strong off the diagonal  (swap hi and lo)
          diagonal (w_rr) = lo        [[.1 .9 .9]
          off-diag (w_rs) = hi         [.9 .1 .9]
                                       [.9 .9 .1]]
          -> w_rs > max(w_rr, w_ss)

      motif="core_periphery"     community 0 is the core; needs hi > mid > lo
          w_00      (core-core)   = hi    [[.9 .5 .5]
          w_0k      (core-peri)   = mid    [.5 .1 .1]
          w_kl      (peri-peri)   = lo     [.5 .1 .1]]
          -> w_rr > w_rs > w_ss for every pair involving community 0.
          NB peripheral pairs (1,2) have all three means equal, so they come
          back "degenerate" -- the paper's rule cannot classify them.

      motif="mixed"              every block drawn U(lo, hi), then symmetrised
          -> pairs land in different classes; needs `rng`.

    So: assortative <-> disassortative is literally swapping hi and lo.
    Core-periphery is the only one that uses `mid`. Raising lo towards hi
    flattens any of them into an Erdos-Renyi reservoir (no meso-scale
    structure), which is the natural null model.

    Args:
        motif: one of MOTIFS
        n_communities: K
        hi/lo: the strong / weak block mean
        mid: intermediate mean, used only for core<->periphery blocks
        rng: required for 'mixed'

    Returns:
        Symmetric (K, K) matrix of block means.
    """
    K = n_communities
    if motif == "assortative":
        omega = np.full((K, K), lo)
        np.fill_diagonal(omega, hi)
    elif motif == "disassortative":
        omega = np.full((K, K), hi)
        np.fill_diagonal(omega, lo)
    elif motif == "core_periphery":
        # community 0 is the core; 1..K-1 are periphery
        omega = np.full((K, K), lo)
        omega[0, :] = omega[:, 0] = mid
        omega[0, 0] = hi
    elif motif == "mixed":
        if rng is None:
            raise ValueError("motif='mixed' needs an rng")
        omega = rng.uniform(lo, hi, size=(K, K))
        omega = (omega + omega.T) / 2  # omega must be symmetric to be classifiable
    else:
        raise ValueError(f"motif must be one of {MOTIFS}, got {motif!r}")
    return omega


def equal_sizes(n_reservoir: int, n_communities: int) -> np.ndarray:
    """Node -> community assignment, communities as equal as possible."""
    return np.repeat(
        np.arange(n_communities),
        np.diff(np.linspace(0, n_reservoir, n_communities + 1).astype(int)),
    )


def dirichlet_sizes(n_reservoir: int, n_communities: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Node -> community assignment with UNEQUAL block sizes.

    Sizes are drawn from Dirichlet(alpha, ..., alpha): small alpha (~1) gives
    strongly lopsided blocks, large alpha (~100) approaches equal. Fitted WSBM
    partitions are never equal-sized, so equal blocks are an assumption worth
    being able to drop.

    Every community gets at least one node, and the sizes sum to n_reservoir.
    """
    K = n_communities
    if K > n_reservoir:
        raise ValueError(f"cannot make {K} communities from {n_reservoir} nodes")
    # one node reserved per community, the rest shared out by the Dirichlet draw
    p = rng.dirichlet(np.full(K, float(alpha)))
    counts = np.ones(K, dtype=int)
    spare = n_reservoir - K
    if spare:
        extra = np.floor(p * spare).astype(int)
        counts += extra
        # hand out the remainder to the largest fractional parts
        short = spare - extra.sum()
        if short:
            order = np.argsort(-(p * spare - extra))
            counts[order[:short]] += 1
    assert counts.sum() == n_reservoir and counts.min() >= 1
    return np.repeat(np.arange(K), counts)


class WSBMESN(BaseESN):
    """Echo state network whose reservoir is drawn from a weighted SBM.

    The architecture is set entirely by `omega` (and the community sizes).
    Pass `motif=` to have the canonical omega built for you, or pass `omega=`
    directly for an arbitrary meso-scale architecture.

    Unlike the rest of this module, this class draws from self.rng (a
    per-instance Generator) rather than the global np.random, so two instances
    built with the same random_state are identical regardless of what else ran
    in between.
    """

    def __init__(
        self,
        n_reservoir: int = 68,
        n_communities: int = 4,
        motif: str = "assortative",
        omega: Optional[np.ndarray] = None,
        assignments: Optional[np.ndarray] = None,
        hi: float = 0.9,
        lo: float = 0.1,
        mid: float = 0.5,
        sigma: float = 0.1,
        connectivity: float = 0.1,
        symmetric: bool = True,
        size_alpha: Optional[float] = None,
        p_negative: float = 0.0,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        ridge_regression_alpha: float = 1e-6,
        random_state: Optional[int] = None,
        input_dim: int = 1,
    ):
        """
        Args:
            n_reservoir: number of reservoir units, N
            n_communities: number of blocks, K (ignored if `assignments` given)
            motif: canonical architecture; ignored if `omega` is given
            omega: explicit (K, K) symmetric block-mean matrix
            assignments: explicit node -> community vector, length N
            hi/lo/mid: block means handed to build_omega
            sigma: within-block weight SD. Scalar, or a symmetric (K, K) matrix
                of per-block SDs -- the paper parameterises each block by both a
                mean and a variance, (mu_rs, sigma^2_rs). Two blockmodels with
                the same omega and different sigma are different models, and the
                motif rule (means only) cannot tell them apart.
            symmetric: draw each undirected edge once and mirror it, so
                W == W.T. True matches the paper (its connectomes are
                undirected). False draws W[i,j] and W[j,i] independently, giving
                complex eigenvalues and richer reservoir dynamics -- not
                paper-faithful, but often what you want for an ESN.
            size_alpha: Dirichlet concentration for community sizes. None (the
                default) makes them equal. Small values (~1) give strongly
                unequal blocks, which is what fitted partitions actually look
                like; large values (~100) approach equal.
            connectivity: Bernoulli edge probability. EITHER a scalar (uniform
                density -- the topology is then Erdos-Renyi and every
                architecture has the same wiring, only different weights) OR a
                symmetric (K, K) matrix of per-block probabilities, which puts
                the meso-scale structure into the topology as well. Build one
                with build_omega(..., hi=p_hi, lo=p_lo) -- it is the same shape.
                Topology and weight structure are independent: an assortative
                `connectivity` with a disassortative `omega` is a legal and
                interesting model.
            p_negative: fraction of edges forced negative. 0.0 = paper-faithful.
                Set >0 only to reproduce older runs -- it biases the realised
                block means away from omega.
            spectral_radius: reservoir is rescaled so |lambda|_max equals this
        """
        super().__init__(
            n_reservoir=n_reservoir,
            input_scaling=input_scaling,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            ridge_regression_alpha=ridge_regression_alpha,
            random_state=random_state,
            input_dim=input_dim,
        )

        if assignments is not None:
            self.community_assignments = np.asarray(assignments)
        elif size_alpha is None:
            self.community_assignments = equal_sizes(n_reservoir, n_communities)
        else:
            self.community_assignments = dirichlet_sizes(n_reservoir, n_communities, size_alpha, self.rng)
        if len(self.community_assignments) != n_reservoir:
            raise ValueError(
                f"assignments has length {len(self.community_assignments)}, expected n_reservoir={n_reservoir}"
            )
        self.n_communities = int(self.community_assignments.max()) + 1

        self.omega = (
            build_omega(motif, self.n_communities, hi, lo, mid, self.rng)
            if omega is None
            else np.asarray(omega, dtype=float)
        )
        if not np.allclose(self.omega, self.omega.T):
            raise ValueError("omega must be symmetric; motif classes are undefined otherwise")

        self.motif = motif
        self.sigma = self._as_block_matrix(sigma, "sigma", lo=0.0)
        self.connectivity = self._as_block_matrix(connectivity, "connectivity", lo=0.0, hi=1.0)
        self.symmetric = symmetric
        self.p_negative = p_negative
        self.initialize_weights()

    def _as_block_matrix(self, value, name, lo=None, hi=None) -> np.ndarray:
        """Accept a scalar or a (K, K) matrix; always store a (K, K) matrix."""
        K = self.n_communities
        m = np.full((K, K), float(value)) if np.isscalar(value) else np.asarray(value, dtype=float)
        if m.shape != (K, K):
            raise ValueError(f"{name} must be a scalar or ({K}, {K}) matrix, got {m.shape}")
        if not np.allclose(m, m.T):
            raise ValueError(f"{name} must be symmetric")
        if (lo is not None and m.min() < lo) or (hi is not None and m.max() > hi):
            raise ValueError(f"{name} must lie in [{lo}, {hi}], got [{m.min():.3g}, {m.max():.3g}]")
        return m

    def initialize_weights(self):
        """Sample input and recurrent weights, then scale the recurrent spectral radius."""
        rng, N = self.rng, self.n_reservoir
        self.W_in = rng.standard_normal((N, self.input_dim)) * self.input_scaling

        # ---- the whole generative model ----
        # omega        -> block mean WEIGHT      mu_rs
        # sigma        -> block weight SD        sigma_rs
        # connectivity -> block edge PROBABILITY theta^e_rs
        # All three are (K, K) and independent, so topology, weight strength and
        # weight dispersion can each carry a different architecture.
        z = self.community_assignments
        W = rng.normal(self.omega[np.ix_(z, z)], self.sigma[np.ix_(z, z)])
        mask = rng.random((N, N)) < self.connectivity[np.ix_(z, z)]

        if self.symmetric:
            # Undirected: draw each edge ONCE (upper triangle) and mirror it.
            # Averaging two independent draws instead would halve the variance
            # and break the sigma_rs parameterisation.
            up = np.triu(np.ones((N, N), dtype=bool), 1)
            W = np.where(up, W, 0.0)
            W = W + W.T
            mask = mask & up
            mask = mask | mask.T

        W *= mask
        np.fill_diagonal(W, 0.0)
        # ------------------------------------

        if self.p_negative > 0:
            flip = rng.random((N, N)) < self.p_negative
            if self.symmetric:
                flip = np.triu(flip, 1)
                flip = flip | flip.T
            W[flip] = -np.abs(W[flip])

        radius = np.abs(np.linalg.eigvals(W)).max()
        if radius > 0:
            W *= self.spectral_radius / radius
        self.W_res = W

    def realised_omega(self) -> np.ndarray:
        """Empirical mu_rs: mean weight of the edges that EXIST in each block.

        Averaged over non-zero entries only, matching the paper -- its weight
        distribution is defined over the set of weighted edges W, not over all
        possible pairs E. Averaging over zeros instead would return
        mu_rs * p_rs, which stops being proportional to omega as soon as
        connectivity varies per block.
        """
        z, K = self.community_assignments, self.n_communities
        out = np.zeros((K, K))
        for r in range(K):
            for s in range(K):
                blk = self.W_res[np.ix_(z == r, z == s)]
                nz = blk[blk != 0]
                out[r, s] = nz.mean() if nz.size else 0.0
        return out

    def realised_sigma(self) -> np.ndarray:
        """Empirical sigma_rs: SD of the edges that exist in each block."""
        z, K = self.community_assignments, self.n_communities
        out = np.zeros((K, K))
        for r in range(K):
            for s in range(K):
                blk = self.W_res[np.ix_(z == r, z == s)]
                nz = blk[blk != 0]
                out[r, s] = nz.std() if nz.size > 1 else 0.0
        return out

    def realised_density(self) -> np.ndarray:
        """Edge density actually present in W_res -- the empirical connectivity."""
        z, K = self.community_assignments, self.n_communities
        return np.array([[(self.W_res[np.ix_(z == r, z == s)] != 0).mean() for s in range(K)] for r in range(K)])

    def motifs(self) -> Dict[Tuple[int, int], str]:
        """Motif class of every community pair, per the paper's rule."""
        return classify_omega(self.omega)

    def get_community_data(self) -> Dict:
        """Return planted assignments, block matrices, and motif classifications."""
        return {
            "community_assignments": self.community_assignments,
            "omega": self.omega,
            "connectivity": self.connectivity,
            "motifs": self.motifs(),
            "density_motifs": classify_omega(self.connectivity),
        }

    def get_node_metadata(self) -> Dict:
        # "core" = the community with the largest within-block mean
        """Return community assignments and a boolean mask identifying core nodes."""
        core = int(np.argmax(np.diag(self.omega)))
        return {
            "community_assignments": self.community_assignments,
            "is_core": self.community_assignments == core,
        }


#######################
# The four architectures -- thin wrappers, one omega each
#######################


def AssortativeESN(n_communities=4, mu_in=0.9, mu_out=0.1, **kw):
    """WSBMESN with strong within-community blocks."""
    return WSBMESN(motif="assortative", n_communities=n_communities, hi=mu_in, lo=mu_out, **kw)


def DisassortativeESN(n_communities=4, mu_in=0.1, mu_out=0.9, **kw):
    """WSBMESN with strong between-community blocks (assortative, hi/lo swapped)."""
    # NB: mu_in is the *within*-block mean, so disassortative == swap hi/lo.
    return WSBMESN(motif="disassortative", n_communities=n_communities, hi=mu_out, lo=mu_in, **kw)


def CorePeripheryESN(n_reservoir=68, n_communities=2, core_fraction=0.2, mu_cc=0.9, mu_cp=0.5, mu_pp=0.1, **kw):
    """WSBMESN with community 0 as a dense core; needs mu_cc > mu_cp > mu_pp."""
    n_core = max(1, int(round(n_reservoir * core_fraction)))
    periphery = equal_sizes(n_reservoir - n_core, max(1, n_communities - 1)) + 1
    assignments = np.concatenate([np.zeros(n_core, dtype=int), periphery])
    return WSBMESN(
        motif="core_periphery",
        n_reservoir=n_reservoir,
        assignments=assignments,
        hi=mu_cc,
        mid=mu_cp,
        lo=mu_pp,
        **kw,
    )


def MixedESN(n_communities=4, hi=0.9, lo=0.1, **kw):
    """WSBMESN with a random symmetric omega -- pairs span several motif classes."""
    return WSBMESN(motif="mixed", n_communities=n_communities, hi=hi, lo=lo, **kw)
