"""Episode-preserving adapters for external DSA DMDc and SubspaceDMDc.

Import after src.analysis.load_dsa. Keep the external estimator and its fitting
settings; construct source/target pairs within trials before concatenating.
"""

import numpy as np
import torch
from DSA.dmdc import DMDc
from DSA.subspace_dmdc import SubspaceDMDc


class EpisodeSeparatedDMDc(DMDc):
    """Use the external DMDc estimator with independent episode transitions."""

    def compute_svd(self):
        """Build X/Y/control inside each episode, then compute the external SVDs."""
        state_arrays = self.H if isinstance(self.H, list) else [self.H]
        input_arrays = self.Hu if isinstance(self.Hu, list) else [self.Hu]
        if len(state_arrays) != len(input_arrays):
            raise ValueError("DMDc state/input episode counts differ")
        episodes, controls = [], []
        for states, inputs in zip(state_arrays, input_arrays):
            if states.ndim == 3:
                episodes.extend(states.unbind(0))
                controls.extend(inputs.unbind(0))
            else:
                episodes.append(states)
                controls.append(inputs)
        if len(episodes) != len(controls) or not episodes:
            raise ValueError("DMDc requires matching nonempty episode and input arrays")
        for states, inputs in zip(episodes, controls):
            if len(states) != len(inputs) or len(states) <= self.steps_ahead:
                raise ValueError("DMDc episode is too short or state/input lengths differ")
        step = self.steps_ahead
        self.episode_transition_counts = [len(states) - step for states in episodes]
        self.X = torch.cat([states[:-step] for states in episodes], dim=0).T
        self.Y = torch.cat([states[step:] for states in episodes], dim=0).T
        control = torch.cat([inputs[:-step] for inputs in controls], dim=0).T
        self.H_shapes = [states.shape for states in episodes]
        self.H = torch.cat(episodes, dim=0)
        self.Hu = torch.cat(controls, dim=0)
        self.Omega = torch.vstack((self.X, control))
        self.Up, self.Sp, vp = torch.linalg.svd(self.Omega, full_matrices=False)
        self.Vp = vp.conj().T
        self.Up1 = self.Up[: self.X.shape[0], :]
        self.Up2 = self.Up[self.X.shape[0] :, :]
        self.Ur, self.Sr, vr = torch.linalg.svd(self.Y, full_matrices=False)
        self.Vr = vr.conj().T
        self.cumulative_explained_variance_input = self._compute_explained_variance(self.Sp)
        self.cumulative_explained_variance_output = self._compute_explained_variance(self.Sr)
        self.episode_boundaries_preserved = True


class EpisodeSeparatedSubspaceDMDc(SubspaceDMDc):
    """Validate every trial and identify N4SID models in float64.

    Keep upstream QR equations and latent realization, with explicit support
    checks before truncation. CPU identification uses NumPy; CUDA uses Torch.
    """

    def __init__(self, data, control_data=None, **kwargs):
        """Normalize identification arrays without altering episode boundaries."""
        if kwargs.get("backend", "n4sid") != "n4sid":
            raise ValueError("Corrected SubspaceDMDc requires the explicit n4sid backend")
        if control_data is None:
            raise ValueError("SubspaceDMDc requires aligned forcing inputs")
        super().__init__(self._double_data(data), control_data=self._double_data(control_data), **kwargs)
        self.device, self.use_torch = self._setup_device(kwargs.get("device", "cpu"))
        self.episode_boundaries_preserved = False

    @staticmethod
    def _double_data(value):
        """Convert array or trial list to finite float64 observation data."""
        if isinstance(value, list):
            return [EpisodeSeparatedSubspaceDMDc._double_data(trial) for trial in value]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim not in (2, 3) or not np.isfinite(array).all():
            raise ValueError("SubspaceDMDc requires finite 2D/3D episode arrays")
        return array

    def fit(self, data=None, control_data=None):
        """Attach the policy identity to numerical and identification failures."""
        try:
            self.episode_boundaries_preserved = False
            if data is not None or control_data is not None:
                if data is None or control_data is None:
                    raise ValueError("N4SID refitting requires both states and forcing inputs")
                self.data, self.control_data = self._double_data(data), self._double_data(control_data)
                data, control_data = self.data, self.control_data
            return super().fit(data, control_data)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
            self.episode_boundaries_preserved = False
            raise ValueError(f"N4SID {getattr(self, 'system_identity', 'system')}: {error}") from error

    def _collect_data(self, y_list, u_list, p, f):
        """Reject unusable episodes rather than silently excluding them upstream."""
        if not y_list or len(y_list) != len(u_list) or p < 1 or f < 1:
            raise ValueError("N4SID requires matching nonempty episodes and positive horizons")
        for i, (y, u) in enumerate(zip(y_list, u_list)):
            if (
                y.ndim != 2 or u.ndim != 2 or len(y) != len(u)
                or y.shape[1] != y_list[0].shape[1] or u.shape[1] != u_list[0].shape[1]
            ):
                raise ValueError(f"N4SID episode {i}: inconsistent state/input dimensions")
            if len(y) - (p + f) < 2:
                raise ValueError(f"N4SID episode {i}: too short for horizons p={p}, f={f} and a transition")
        return super()._collect_data(y_list, u_list, p, f)

    def _time_align_valid_trials(self, X_hat, u_list, y_list, valid_trials, T_per_trial, p):
        """Align latent transitions to source forcing separately within each trial."""
        if valid_trials != list(range(len(y_list))) or any(t < 2 for t in T_per_trial):
            raise ValueError("N4SID must use every episode with at least one transition")
        result = super()._time_align_valid_trials(X_hat, u_list, y_list, valid_trials, T_per_trial, p)
        self.episode_transition_counts = [t - 1 for t in T_per_trial]
        if result[0].shape[1] != sum(self.episode_transition_counts):
            raise ValueError("N4SID transition counts disagree with trial alignment")
        self.episode_boundaries_preserved = True
        return result

    def subspace_dmdc_multitrial_QR_decomposition(self, y_list, u_list, p, f, n=None, lamb=1e-8, energy=0.999):
        """Apply upstream QR identification with regression-block and numerical-rank checks."""
        U_p, Y_p, U_f, Y_f, Z_p, valid, counts, total, output_dim, input_dim = self._collect_data(
            y_list, u_list, p, f
        )
        H = np.vstack([U_f, Z_p, Y_f])
        dim_uf, dim_zp = f * input_dim, p * (input_dim + output_dim)
        # Reduced QR can have fewer columns than H has rows. Only the columns
        # spanning U_f and Z_p are needed for R22 and R32; Y_f adds rows only.
        required_columns = dim_uf + dim_zp
        if H.shape[1] < required_columns:
            raise ValueError(
                f"N4SID incomplete QR blocks: {H.shape[1]} windows for "
                f"{required_columns} required future-input/past-data columns ({H.shape[0]} total rows)"
            )
        if self.use_torch:
            h, zp = self._to_torch(H), self._to_torch(Z_p)
            _, r = torch.linalg.qr(h.T, mode="reduced")
            lower = r.T
            r22 = lower[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
            r32 = lower[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
            inverse = torch.linalg.solve(
                r22.T @ r22 + lamb * torch.eye(dim_zp, device=r22.device, dtype=torch.float64), r22.T
            )
            projection = r32 @ inverse @ zp
            uo, singular, vt = torch.linalg.svd(projection, full_matrices=False)
            uo, singular, vt = (self._to_numpy(x) for x in (uo, singular, vt))
        else:
            _, r = np.linalg.qr(H.T, mode="reduced")
            lower = r.T
            r22 = lower[dim_uf:dim_uf + dim_zp, dim_uf:dim_uf + dim_zp]
            r32 = lower[dim_uf + dim_zp:, dim_uf:dim_uf + dim_zp]
            inverse = np.linalg.solve(r22.T @ r22 + lamb * np.eye(dim_zp), r22.T)
            projection = r32 @ inverse @ Z_p
            uo, singular, vt = np.linalg.svd(projection, full_matrices=False)
        if not np.isfinite(singular).all() or not singular.size or singular[0] <= 0:
            raise ValueError("N4SID identification projection has zero energy or nonfinite singular values")
        tolerance = max(projection.shape) * np.finfo(np.float64).eps * singular[0]
        support = int(np.count_nonzero(singular > tolerance))
        if n is None:
            n = int(np.searchsorted(np.cumsum(singular**2) / np.sum(singular**2), energy)) + 1
        if not 1 <= n <= support:
            raise ValueError(f"N4SID requested rank {n} exceeds projection numerical support {support}")
        root = np.diag(np.sqrt(singular[:n]))
        gamma, latent = uo[:, :n] @ root, root @ vt[:n]
        X, X_next, U_mid, Y_curr = self._time_align_valid_trials(latent, u_list, y_list, valid, counts, p)
        a, b = self._perform_ridge_regression(X, X_next, U_mid, n, lamb)
        c = gamma[:output_dim]
        noise, r_noise, q_noise, s_noise = self._estimate_noise_covariance(X_next, a, X, b, U_mid, Y_curr, c)
        if not all(np.isfinite(x).all() for x in (a, b, c, noise)):
            raise ValueError("N4SID produced nonfinite identified operators or noise covariance")
        if a.shape != (n, n) or b.shape != (n, input_dim) or c.shape != (output_dim, n):
            raise ValueError("N4SID identified operator dimensions disagree with requested order")
        self.identification_diagnostics = dict(
            identification_dtype="float64", past_horizon=p, future_horizon=f,
            projection_singular_values=singular.tolist(), projection_support=support,
            projection_support_tolerance=float(tolerance), projection_shape=list(projection.shape),
            projection_condition=float(singular[0] / singular[n - 1]),
            episodes_used=len(valid), episode_window_counts=counts,
            state_operator_shape=list(a.shape), control_operator_shape=list(b.shape),
            observation_operator_shape=list(c.shape), qr_condition=float(np.linalg.cond(self._to_numpy(r22))),
        )
        info = dict(
            singular_values_O=singular, rank_used=n, Gamma_hat=gamma, f=f,
            n_trials_total=len(y_list), n_trials_used=len(valid), valid_trials=valid,
            T_per_trial=counts, T_total=total, trial_lengths=[len(y) for y in y_list],
            noise_covariance=noise, R_hat=r_noise, Q_hat=q_noise, S_hat=s_noise, X_hat=latent,
        )
        return a, b, c, info
