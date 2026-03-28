"""
Experiment 2 Model Predictions

Uses parameters fitted to Experiment 1 (from fit.py) to generate predictions
for Experiment 2. Simulates full 8-round belief trajectories.

Models (base models without response weights)
------
- Belief model (α): Simulates slider condition (belief averaging)
- Evidence model (δ, α): Simulates chat condition (evidence pooling)

Simulation
----------
For each Exp2 game:
1. Initialize beliefs at each player's individual MLE
2. Simulate 8 rounds of belief updates with partner interactions
3. Add Gaussian noise N(0, σ) at each round (from fitting residuals)
4. Compute final prediction error vs network MLE

Output
------
- figures/model_fig5.pdf: Predicted error across difficulty axes
- Console: Means and effects by representativeness

Usage
-----
  python predict.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# LOAD FITTED PARAMETERS FROM fit.py (base models, no response weights)
# =============================================================================

fit_results = pd.read_csv('fit_results.csv')

# Base Belief model: slider, degroot, anchor=False (1 param: α)
belief_fit = fit_results[(fit_results['condition'] == 'slider') &
                         (fit_results['model'] == 'degroot') &
                         (fit_results['anchor'] == False)].iloc[0]

# Base Evidence model: chat, count, anchor=False (2 params: δ, α)
evidence_fit = fit_results[(fit_results['condition'] == 'chat') &
                           (fit_results['model'] == 'count') &
                           (fit_results['anchor'] == False)].iloc[0]

BELIEF_ALPHA = belief_fit['alpha']
BELIEF_SIGMA = belief_fit['sigma']

EVIDENCE_DELTA = evidence_fit['delta']
EVIDENCE_ALPHA = evidence_fit['alpha']
EVIDENCE_SIGMA = evidence_fit['sigma']

print("=" * 70)
print("EXPERIMENT 2: MODEL PREDICTIONS (base models)")
print("=" * 70)
print(f"\nLoaded parameters from fit_results.csv:")
print(f"  Belief (Slider):   α={BELIEF_ALPHA:.2f}, σ={BELIEF_SIGMA:.3f}")
print(f"  Evidence (Chat):   δ={EVIDENCE_DELTA:.2f}, α={EVIDENCE_ALPHA:.2f}, σ={EVIDENCE_SIGMA:.3f}")

# =============================================================================
# SETUP
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

EVIDENCE_COLOR = '#2E8B57'    # Sea green (chat)
BELIEF_COLOR = '#DAA520'      # Goldenrod (slider)


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

exp2 = pd.read_csv('../analysis/exp2/exp2_clean_data.csv')

# Derive condition and direction from treatmentName
treat_col = 'treatmentName.x' if 'treatmentName.x' in exp2.columns else 'treatmentName'
exp2['condition'] = exp2[treat_col].apply(lambda x: 'interactive' if 'interactive' in str(x) else 'slider')
exp2['direction'] = exp2[treat_col].apply(lambda x: 'RS' if str(x).startswith('RS_') else 'SR')

print(f"Loaded Exp 2: {exp2['gameID'].nunique()} games")

# Get game-level data
game_data = exp2.groupby('gameID').first().reset_index()
game_data = game_data[['gameID', 'condition', 'direction', 'nCrittersGame',
                        'nRabbitsGame', 'entropy', 'representation']]

# Network MLE aligned for direction
game_data['network_mle_raw'] = game_data['nRabbitsGame'] / game_data['nCrittersGame']
game_data['network_mle_aligned'] = np.where(
    game_data['direction'] == 'RS',
    1 - game_data['network_mle_raw'],
    game_data['network_mle_raw']
)

# Representativeness bins (median split)
rep_median = game_data['representation'].median()
game_data['rep_bin'] = np.where(
    game_data['representation'] <= rep_median, 'Representative', 'Unrepresentative'
)


# =============================================================================
# MODEL SIMULATIONS
# =============================================================================

def simulate_belief(game_df, alpha=BELIEF_ALPHA, sigma=BELIEF_SIGMA,
                    n_rounds=8, n_sims=1000):
    """
    Belief model: agents average their belief with their partner's belief.

    belief(t+1) = α * own_belief + (1-α) * partner_belief + N(0, σ)
    """
    players = game_df.drop_duplicates('playerID')
    n_players = len(players)

    if n_players < 2:
        return None

    player_mles = []
    for _, row in players.iterrows():
        mle = row['nRabbits'] / row['nCritters'] if row['nCritters'] > 0 else 0.5
        player_mles.append(mle)

    all_final_beliefs = []
    for _ in range(n_sims):
        beliefs = list(player_mles)

        for round_idx in range(n_rounds):
            new_beliefs = []
            for p_idx in range(n_players):
                other_players = [i for i in range(n_players) if i != p_idx]
                partner_idx = other_players[round_idx % len(other_players)]

                new_belief = alpha * beliefs[p_idx] + (1 - alpha) * beliefs[partner_idx]
                new_belief = np.clip(new_belief + np.random.normal(0, sigma), 0, 1)
                new_beliefs.append(new_belief)

            beliefs = new_beliefs

        all_final_beliefs.append(beliefs)

    return np.mean(all_final_beliefs, axis=0)


def simulate_evidence(game_df, delta=EVIDENCE_DELTA, alpha=EVIDENCE_ALPHA,
                      sigma=EVIDENCE_SIGMA, n_rounds=8, n_sims=1000):
    """
    Evidence model: agents pool sample counts to compute a pooled MLE.

    pooled_mle = (own_r + δ * cum_r) / (own_n + δ * cum_n)
    belief(t+1) = α * own_belief + (1-α) * pooled_mle + N(0, σ)
    """
    players = game_df.drop_duplicates('playerID')
    n_players = len(players)

    if n_players < 2:
        return None

    player_list = []
    for _, row in players.iterrows():
        n = int(row['nCritters'])
        player_list.append({
            'r': int(row['nRabbits']),
            's': int(row['nSquirrels']),
            'n': n,
            'mle': row['nRabbits'] / n if n > 0 else 0.5
        })

    # Precompute cumulative evidence for each player (from all others)
    player_cum_r = []
    player_cum_s = []
    for p_idx in range(n_players):
        cum_r = sum(p['r'] for i, p in enumerate(player_list) if i != p_idx)
        cum_s = sum(p['s'] for i, p in enumerate(player_list) if i != p_idx)
        player_cum_r.append(cum_r)
        player_cum_s.append(cum_s)

    all_final_beliefs = []
    for _ in range(n_sims):
        beliefs = [p['mle'] for p in player_list]

        for round_idx in range(n_rounds):
            new_beliefs = []
            for p_idx, player in enumerate(player_list):
                # Pooled MLE (own + discounted partner evidence)
                pooled_r = player['r'] + delta * player_cum_r[p_idx]
                pooled_n = player['n'] + delta * (player_cum_r[p_idx] + player_cum_s[p_idx])
                pooled_mle = pooled_r / pooled_n if pooled_n > 0 else 0.5

                # Self-weight step
                new_belief = alpha * beliefs[p_idx] + (1 - alpha) * pooled_mle

                # Add noise and clip
                new_belief = np.clip(new_belief + np.random.normal(0, sigma), 0, 1)
                new_beliefs.append(new_belief)

            beliefs = new_beliefs

        all_final_beliefs.append(beliefs)

    return np.mean(all_final_beliefs, axis=0)


# =============================================================================
# COMPUTE PREDICTIONS
# =============================================================================

np.random.seed(42)
print("\nComputing model predictions...")

model_errors = []
for _, row in game_data.iterrows():
    game_id = row['gameID']
    game_df = exp2[exp2['gameID'] == game_id]
    round1 = game_df[game_df['idx'] == 1]

    if len(round1) < 2:
        continue

    direction = row['direction']
    network_mle_aligned = row['network_mle_aligned']

    # Belief model (slider)
    belief_beliefs = simulate_belief(round1)
    if belief_beliefs is None:
        continue
    if direction == 'RS':
        belief_beliefs = 1 - belief_beliefs
    belief_error = np.abs(belief_beliefs.mean() - network_mle_aligned) * 100

    # Evidence model (chat)
    evidence_beliefs = simulate_evidence(round1)
    if evidence_beliefs is None:
        continue
    if direction == 'RS':
        evidence_beliefs = 1 - evidence_beliefs
    evidence_error = np.abs(evidence_beliefs.mean() - network_mle_aligned) * 100

    model_errors.append({
        'gameID': game_id,
        'belief_error': belief_error,
        'evidence_error': evidence_error,
        'rep_bin': row['rep_bin']
    })

model_df = pd.DataFrame(model_errors)
print(f"Model predictions: {len(model_df)} games")


# =============================================================================
# COMPUTE AND PRINT RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("PREDICTIONS BY REPRESENTATIVENESS")
print("=" * 70)

for model_name, col in [('Belief', 'belief_error'), ('Evidence', 'evidence_error')]:
    rep = model_df[model_df['rep_bin'] == 'Representative'][col]
    unrep = model_df[model_df['rep_bin'] == 'Unrepresentative'][col]
    print(f"\n{model_name} model:")
    print(f"  Representative:   {rep.mean():.1f} (SD={rep.std():.1f})")
    print(f"  Unrepresentative: {unrep.mean():.1f} (SD={unrep.std():.1f})")
    print(f"  Effect:           {unrep.mean() - rep.mean():.1f}")

belief_effect = (model_df[model_df['rep_bin'] == 'Unrepresentative']['belief_error'].mean() -
                 model_df[model_df['rep_bin'] == 'Representative']['belief_error'].mean())
evidence_effect = (model_df[model_df['rep_bin'] == 'Unrepresentative']['evidence_error'].mean() -
                   model_df[model_df['rep_bin'] == 'Representative']['evidence_error'].mean())
print(f"\nRatio of effects: {belief_effect / evidence_effect:.1f}x")


# =============================================================================
# CREATE FIGURE (matching Figure 5B layout)
# =============================================================================

print("\nCreating figure...")

fig, ax = plt.subplots(figsize=(4.5, 4))

x = np.array([0, 1])
# Unrepresentative on left, Representative on right (matching Fig 5B)
labels = ['Unrepresentative', 'Representative']

# Compute means and bootstrap CIs
def bootstrap_ci(values, n_boot=1000, seed=42):
    np.random.seed(seed)
    means = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])

belief_means = []
belief_cis = []
evidence_means = []
evidence_cis = []

for rep_level in labels:
    b_vals = model_df[model_df['rep_bin'] == rep_level]['belief_error'].values
    e_vals = model_df[model_df['rep_bin'] == rep_level]['evidence_error'].values

    belief_means.append(np.mean(b_vals))
    belief_cis.append(bootstrap_ci(b_vals))
    evidence_means.append(np.mean(e_vals))
    evidence_cis.append(bootstrap_ci(e_vals))

# Error bars
belief_errs = [[belief_means[i] - belief_cis[i][0] for i in range(2)],
               [belief_cis[i][1] - belief_means[i] for i in range(2)]]
evidence_errs = [[evidence_means[i] - evidence_cis[i][0] for i in range(2)],
                 [evidence_cis[i][1] - evidence_means[i] for i in range(2)]]

# Connected lines with error bars (matching Fig 5B style)
ax.errorbar(x, evidence_means, yerr=evidence_errs,
            color=EVIDENCE_COLOR, marker='o', markersize=8, linewidth=2, capsize=3,
            label='Evidence (chat)')
ax.errorbar(x, belief_means, yerr=belief_errs,
            color=BELIEF_COLOR, marker='o', markersize=8, linewidth=2, capsize=3,
            label='Belief (slider)')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Predicted Error')
ax.set_title('Representativeness')
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(0, None)
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('../figures/fig6.pdf', bbox_inches='tight')
plt.savefig('../figures/fig6.png', bbox_inches='tight', dpi=300)
print("Saved to figures/fig6.pdf")
