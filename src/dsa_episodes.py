"""Episode-preserving adapter for the external DSA DMDc implementation.

Import after src.analysis.load_dsa. Keep the external estimator and its fitting
settings; construct source/target pairs within trials before concatenating.
"""

import torch
from DSA.dmdc import DMDc


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
