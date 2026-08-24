import numpy as np

def choose_rank(Ys, n_delays, energy=0.99, min_rank=1, max_rank=50):
    """Pick the DMD rank from the data instead of hard-coding it.

    Returns the smallest rank at which EVERY system reaches `energy` of its
    cumulative singular energy.
    """
    per_system = []
    for Y in Ys:
        # Avoid empty arrays if the sequence is shorter than n_delays
        if len(Y) < n_delays:
            continue
        H = np.hstack([Y[i : len(Y) - n_delays + i + 1] for i in range(n_delays)])
        sv = np.linalg.svd(H - H.mean(axis=0), compute_uv=False)
        cum = np.cumsum(sv**2) / np.sum(sv**2)
        per_system.append(int(np.searchsorted(cum, energy)) + 1)

    if not per_system:
        return min_rank, []

    rank = int(np.clip(max(per_system), min_rank, max_rank))
    return rank, per_system
