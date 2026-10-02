"""Numerically safe evaluation of the external DSA controllability metric.

Preserve DSA's controllability blocks, alignment orientation, and raw distance
units. Only calculation precision and the scale of the SVD input change.
Import this module after loading the external DSA checkout.
"""

import numpy as np
from DSA.simdist_controllability import ControllabilitySimilarityTransformDist
from scipy.linalg import norm


class StableControllabilityDistance(ControllabilitySimilarityTransformDist):
    """Evaluate DSA's existing metric in float64 with a bounded SVD input."""

    def fit_score(self, A, B, A_control, B_control):
        """Promote fitted operators before constructing any matrix powers."""
        matrices = []
        for value in (A, B, A_control, B_control):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError("InputDSA received nonfinite fitted dynamics operators")
            matrices.append(value)
        result = super().fit_score(*matrices)
        if not np.isfinite(result).all():
            raise ValueError("InputDSA distance exceeds float64 range; inspect fitted dynamics and rank")
        return result

    def get_controllability_matrix(self, A, B):
        """Retain external DSA's blocks, rejecting unrepresentable matrix powers."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                return super().get_controllability_matrix(
                    np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
                )
        except FloatingPointError as error:
            raise ValueError(
                "InputDSA controllability powers exceed float64 range; "
                "inspect fitted dynamics, reduce rank, or increase DMD regularization explicitly"
            ) from error

    def compare_systems_procrustes(self, A1, B1, A2, B2, *, align_inputs=False):
        """Scale the cross-product without changing its orthogonal alignment."""
        if align_inputs:
            raise ValueError("Input alignment is unsupported by the external DSA metric")
        K1 = self.get_controllability_matrix(A1, B1)
        K2 = self.get_controllability_matrix(A2, B2)
        # Independent positive scalar factors preserve the SVD's U @ Vh.
        scale1 = max(float(np.max(np.abs(K1))), 1.0)
        scale2 = max(float(np.max(np.abs(K2))), 1.0)
        bounded1, bounded2 = K1 / scale1, K2 / scale2
        U, _, Vh = np.linalg.svd(bounded2 @ bounded1.T, full_matrices=False)
        C = U @ Vh
        # Keep the external implementation's orientation and original units.
        aligned2 = C @ K2
        error = norm((K1 - aligned2).ravel())
        aligned_bounded2 = C @ bounded2
        norm1, norm2 = norm(bounded1.ravel()), norm(aligned_bounded2.ravel())
        if norm1 == 0 or norm2 == 0:
            angular = 0.0 if norm1 == norm2 else np.pi / 2
        else:
            cosine = np.vdot(bounded1 / norm1, aligned_bounded2 / norm2).real
            angular = np.arccos(np.clip(cosine, -1, 1))
        return C, np.eye(B2.shape[-1]), error, angular
