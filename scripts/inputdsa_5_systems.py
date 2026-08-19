import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import DSA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DSA')))
try:
    from DSA import InputDSA
except ImportError:
    print("Could not import DSA. Ensure it is accessible in the parent directory.")
    sys.exit(1)

from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes.wsbm_esn import AssortativeESN, DisassortativeESN, CorePeripheryESN, MixedESN

def parse_selected_csv(csv_path="selected.csv"):
    df = pd.read_csv(csv_path)
    # The 'null' motif is parsed as NaN by pandas
    df['stratum'] = df['stratum'].fillna('null')

    # Get first instance of each stratum
    first_instances = df.groupby('stratum').first().reset_index()

    configs = []
    for _, row in first_instances.iterrows():
        motif = row['stratum']
        n_communities = int(row['K'])
        connectivity = row['p0']
        centre_u = row['centre_u']
        contrast_u = row['contrast_u']
        sigma_ratio = row['sigma_ratio']
        p_negative = row['frac_negative']
        seed = int(row['seed'])
        core_fraction = row.get('core_fraction', 0.2)

        # In wsbm_esn, contrast is typically max_u - min_u
        hi = centre_u + (contrast_u / 2)
        lo = centre_u - (contrast_u / 2)
        mid = centre_u
        sigma = sigma_ratio * centre_u

        configs.append({
            "motif": motif,
            "n_communities": n_communities,
            "hi": hi,
            "lo": lo,
            "mid": mid,
            "sigma": sigma,
            "connectivity": connectivity,
            "p_negative": p_negative,
            "seed": seed,
            "core_fraction": core_fraction
        })
    return configs

def main():
    configs = parse_selected_csv("selected.csv")

    systems = []
    labels = []

    units = 100

    print("Initializing ESN models...")
    for config in configs:
        motif = config['motif']

        common_kwargs = {
            'n_communities': config['n_communities'],
            'connectivity': config['connectivity'],
            'sigma': config['sigma'],
            'p_negative': config['p_negative'],
            'seed': config['seed']
        }

        if motif == 'assortative':
            esn = AssortativeESN(mu_in=config['hi'], mu_out=config['lo'], units=units, **common_kwargs)
        elif motif == 'disassortative':
            esn = DisassortativeESN(mu_in=config['hi'], mu_out=config['lo'], units=units, **common_kwargs)
        elif motif == 'core_periphery':
            esn = CorePeripheryESN(
                n_reservoir=units,
                mu_cc=config['hi'],
                mu_cp=config['mid'],
                mu_pp=config['lo'],
                core_fraction=config['core_fraction'],
                **common_kwargs
            )
        elif motif == 'mixed':
            esn = MixedESN(hi=config['hi'], lo=config['lo'], units=units, **common_kwargs)
        elif motif == 'null':
            # Use AssortativeESN with equal means for Erdős-Rényi graph (null model)
            esn = AssortativeESN(mu_in=config['mid'], mu_out=config['mid'], units=units, **common_kwargs)
        else:
            print(f"Unknown motif: {motif}")
            continue

        systems.append(esn)
        labels.append(motif)
        print(f"  Initialized {motif} ESN.")

    print(f"\nGenerated {len(systems)} systems: {labels}")

    # Generate Mackey-Glass time series
    N = 2000
    print(f"Generating Mackey-Glass timeseries (N={N})...")
    mg_data = mackey_glass(n_timesteps=N)
    U = mg_data.reshape(-1, 1)

    # Run inputs through systems
    Ys = []
    Us = []
    print("Running timeseries through systems...")
    for label, esn in zip(labels, systems):
        print(f"  Running {label}...")
        # esn.run returns the state matrix of shape (N, units)
        Y = esn.run(U)
        Ys.append(Y)
        Us.append(U)

    # InputDSA Analysis
    dmd_config = dict(
        n_delays=10,
        rank=20,
        backend='n4sid'
    )

    print("\nFitting InputDSA...")
    inputDSA = InputDSA(
        X=Ys,
        X_control=Us,
        dmd_config=dmd_config,
        simdist_config={'compare': 'joint', 'return_distance_components': True}
        
    )

    res = inputDSA.fit_score()
    sims_full = res[:, :, 0]
    sims_state_joint = res[:, :, 1]
    sims_control_joint = res[:, :, 2]

    inputDSA.update_compare_method(compare='state')
    sims_state_separate = inputDSA.score()

    inputDSA.update_compare_method(compare='control', simdist_config={'score_method': 'euclidean'})
    sims_control_separate = inputDSA.score()

    print("\nPlotting results...")
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    sims_data = [sims_full, sims_state_joint, sims_control_joint, sims_state_separate, sims_control_separate]
    titles = ['Joint', 'State (Joint)', 'Control (Joint)', 'State (Separate)', 'Control (Separate)']

    for i, (data, title) in enumerate(zip(sims_data, titles)):
        im = ax[i].imshow(data, cmap='viridis')
        cbar = plt.colorbar(im, ax=ax[i], shrink=0.7, location='top')
        ax[i].set_title(title, y=1.2, pad=10)
        ax[i].set_xticks(range(len(labels)))
        ax[i].set_yticks(range(len(labels)))
        ax[i].set_xticklabels(labels, rotation=45, ha='right')
        ax[i].set_yticklabels(labels)

    plt.tight_layout()

    out_file = "inputdsa_5_systems_results.png"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
