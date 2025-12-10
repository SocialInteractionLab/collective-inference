"""
Computational Models of Social Learning - Experiment 1 Model Fitting

This is the single entry point for fitting all models to Experiment 1 data.

Models (structurally matched with anchoring in both):
  1. DeGroot + Anchoring: Belief averaging with self-weight (α) and anchoring (β)
  2. Count + Relative Anchoring: Evidence pooling with discount (δ), inertia (γ),
     and sample-size-dependent anchoring (β_scale)

The key theoretical distinction:
  - DeGroot uses partner's BELIEF (available in slider condition)
  - Count uses partner's COUNTS (available in chat condition)

Both models include anchoring to own MLE, but Count's anchoring is "relative":
  β_i = β_scale × (1 - relative_sample_size × n_players)
  Players with smaller samples anchor MORE to their own (noisy) MLE.

Fitting approach:
  TRANSITION-BASED: Fits round-to-round belief transitions.
  This directly tests each model's update rule against human behavior.

Usage:
  python fit.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple


# =============================================================================
# DATA LOADING
# =============================================================================

def load_exp1_data():
    """Load Experiment 1 belief data and partner assignments."""
    beliefs = pd.read_csv('../analysis/exp1/exp1_clean_data.csv')
    beliefs['belief'] = beliefs['guess'] / 100.0
    beliefs['nCritters'] = beliefs['nRabbits'] + beliefs['nSquirrels']

    # Load partner assignments from messages
    msgs = pd.read_csv('../data/exp1/raw/1_extracted_msgs_prereg/extracted_messages.csv')
    msgs = msgs.dropna(subset=['round', 'recipient', 'sender.id', 'groupID'])

    partner_lookup = {}
    for _, row in msgs.iterrows():
        key = (row['groupID'], row['recipient'], int(row['round']))
        if key not in partner_lookup:
            partner_lookup[key] = row['sender.id']

    return beliefs, partner_lookup


def get_player_samples(beliefs: pd.DataFrame) -> Dict[Tuple, Tuple[int, int]]:
    """Extract each player's sample counts."""
    samples = {}
    for _, row in beliefs.drop_duplicates(['gameID', 'playerID']).iterrows():
        key = (row['gameID'], row['playerID'])
        samples[key] = (int(row['nRabbits']), int(row['nSquirrels']))
    return samples


# =============================================================================
# PREPARE TRANSITION DATA
# =============================================================================

def prepare_transitions(beliefs: pd.DataFrame,
                        partner_lookup: Dict,
                        samples: Dict,
                        condition: str) -> pd.DataFrame:
    """
    Extract all belief transitions with context needed for model fitting.
    """
    cond_data = beliefs[beliefs['condition'] == condition]
    transitions = []

    for game_id in cond_data['gameID'].unique():
        game_df = cond_data[cond_data['gameID'] == game_id]

        # Compute total sample size for this game (for relative anchoring)
        game_total_n = 0
        game_players = game_df['playerID'].unique()
        n_players = len(game_players)
        for pid in game_players:
            r, s = samples.get((game_id, pid), (0, 0))
            game_total_n += r + s

        for player_id in game_df['playerID'].unique():
            player_df = game_df[game_df['playerID'] == player_id].sort_values('idx')
            rounds = player_df['idx'].values
            player_beliefs = player_df['belief'].values

            own_r, own_s = samples.get((game_id, player_id), (0, 0))
            own_n = own_r + own_s
            own_mle = own_r / own_n if own_n > 0 else 0.5

            # Relative sample size for this player
            relative_size = own_n / game_total_n if game_total_n > 0 else 1/n_players

            # Track cumulative evidence from unique partners
            seen_partners = set()
            cumulative_r, cumulative_s = 0, 0

            for t_idx in range(len(rounds) - 1):
                t = int(rounds[t_idx])
                own_belief = player_beliefs[t_idx]
                actual_next = player_beliefs[t_idx + 1]

                # Get partner info
                partner_key = (game_id, player_id, t)
                partner_id = partner_lookup.get(partner_key)
                if partner_id is None:
                    continue

                partner_sample = samples.get((game_id, partner_id))
                if partner_sample is None:
                    continue
                partner_r, partner_s = partner_sample
                partner_n = partner_r + partner_s
                partner_mle = partner_r / partner_n if partner_n > 0 else 0.5

                # Get partner's belief at time t
                partner_df = game_df[game_df['playerID'] == partner_id]
                partner_at_t = partner_df[partner_df['idx'] == t]
                if len(partner_at_t) == 0:
                    continue
                partner_belief = partner_at_t['belief'].values[0]

                # Update cumulative evidence (only count new partners)
                if partner_id not in seen_partners:
                    seen_partners.add(partner_id)
                    cumulative_r += partner_r
                    cumulative_s += partner_s

                transitions.append({
                    'game_id': game_id,
                    'player_id': player_id,
                    'round': t,
                    'own_belief': own_belief,
                    'actual_next': actual_next,
                    'own_r': own_r,
                    'own_s': own_s,
                    'own_mle': own_mle,
                    'partner_belief': partner_belief,
                    'partner_mle': partner_mle,
                    'cumulative_r': cumulative_r,
                    'cumulative_s': cumulative_s,
                    'relative_size': relative_size,
                    'n_players': n_players
                })

    return pd.DataFrame(transitions)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

@dataclass
class DeGrootParams:
    """DeGroot + Anchoring: 2 parameters."""
    alpha: float  # Self-weight in DeGroot averaging
    beta: float   # Anchor strength toward own MLE


@dataclass
class CountParams:
    """Count + Relative Anchoring: 3 parameters."""
    delta: float      # Evidence discount for others' counts
    gamma: float      # Update inertia
    beta_scale: float # Anchoring scale (β = β_scale × (1 - relative_size × n))


# =============================================================================
# MODEL PREDICTIONS
# =============================================================================

def degroot_predict(row, params: DeGrootParams) -> float:
    """
    DeGroot + Anchoring prediction.

    Update rule:
        degroot_step = α × own_belief + (1-α) × partner_belief
        prediction = β × own_mle + (1-β) × degroot_step
    """
    degroot_step = params.alpha * row['own_belief'] + (1 - params.alpha) * row['partner_belief']
    return params.beta * row['own_mle'] + (1 - params.beta) * degroot_step


def count_predict(row, params: CountParams) -> float:
    """
    Count + Relative Anchoring prediction.

    Update rule:
        pooled_mle = (own_r + δ × cumulative_r) / (own_n + δ × cumulative_n)
        pooled_step = γ × own_belief + (1-γ) × pooled_mle
        β_i = β_scale × (1 - relative_size × n_players)
        prediction = β_i × own_mle + (1-β_i) × pooled_step

    Players with smaller relative samples anchor MORE to their own MLE.
    """
    own_n = row['own_r'] + row['own_s']
    cumulative_n = row['cumulative_r'] + row['cumulative_s']

    # Pooled MLE with discounted evidence
    pooled_r = row['own_r'] + params.delta * row['cumulative_r']
    pooled_n = own_n + params.delta * cumulative_n
    pooled_mle = pooled_r / pooled_n if pooled_n > 0 else 0.5

    # Inertia step
    pooled_step = params.gamma * row['own_belief'] + (1 - params.gamma) * pooled_mle

    # Relative anchoring: smaller samples → stronger anchoring
    beta = params.beta_scale * (1 - row['relative_size'] * row['n_players'])
    beta = np.clip(beta, 0, 0.9)

    return beta * row['own_mle'] + (1 - beta) * pooled_step


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_degroot(transitions: pd.DataFrame, with_anchor: bool = True) -> Tuple[DeGrootParams, float]:
    """Fit DeGroot model to transitions."""
    if with_anchor:
        def objective(params_array):
            alpha, beta = params_array
            if not (0 < alpha < 1 and 0 < beta < 1):
                return 1e10
            params = DeGrootParams(alpha=alpha, beta=beta)
            preds = transitions.apply(lambda r: degroot_predict(r, params), axis=1)
            return np.mean((preds - transitions['actual_next']) ** 2)

        best_result, best_mse = None, np.inf
        for a0 in [0.6, 0.75, 0.85]:
            for b0 in [0.1, 0.2, 0.3]:
                result = minimize(objective, x0=[a0, b0],
                                bounds=[(0.01, 0.99), (0.01, 0.99)],
                                method='L-BFGS-B')
                if result.fun < best_mse:
                    best_mse = result.fun
                    best_result = result
        return DeGrootParams(alpha=best_result.x[0], beta=best_result.x[1]), best_mse
    else:
        # Ablated: no anchoring (β=0)
        def objective(params_array):
            alpha = params_array[0]
            if not (0 < alpha < 1):
                return 1e10
            params = DeGrootParams(alpha=alpha, beta=0.0)
            preds = transitions.apply(lambda r: degroot_predict(r, params), axis=1)
            return np.mean((preds - transitions['actual_next']) ** 2)

        best_result, best_mse = None, np.inf
        for a0 in [0.6, 0.75, 0.85]:
            result = minimize(objective, x0=[a0],
                            bounds=[(0.01, 0.99)],
                            method='L-BFGS-B')
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result
        return DeGrootParams(alpha=best_result.x[0], beta=0.0), best_mse


def fit_count(transitions: pd.DataFrame, with_anchor: bool = True) -> Tuple[CountParams, float]:
    """Fit Count model to transitions."""
    if with_anchor:
        def objective(params_array):
            delta, gamma, beta_scale = params_array
            if not (0 < delta < 3 and 0 < gamma < 1 and 0 <= beta_scale < 2):
                return 1e10
            params = CountParams(delta=delta, gamma=gamma, beta_scale=beta_scale)
            preds = transitions.apply(lambda r: count_predict(r, params), axis=1)
            return np.mean((preds - transitions['actual_next']) ** 2)

        best_result, best_mse = None, np.inf
        for d0 in [0.3, 0.5, 0.7]:
            for g0 in [0.3, 0.5, 0.7]:
                for bs0 in [0.1, 0.3, 0.5]:
                    result = minimize(objective, x0=[d0, g0, bs0],
                                    bounds=[(0.01, 2.0), (0.01, 0.99), (0.0, 1.5)],
                                    method='L-BFGS-B')
                    if result.fun < best_mse:
                        best_mse = result.fun
                        best_result = result
        return CountParams(delta=best_result.x[0], gamma=best_result.x[1],
                           beta_scale=best_result.x[2]), best_mse
    else:
        # Ablated: no relative anchoring (β_scale=0)
        def objective(params_array):
            delta, gamma = params_array
            if not (0 < delta < 3 and 0 < gamma < 1):
                return 1e10
            params = CountParams(delta=delta, gamma=gamma, beta_scale=0.0)
            preds = transitions.apply(lambda r: count_predict(r, params), axis=1)
            return np.mean((preds - transitions['actual_next']) ** 2)

        best_result, best_mse = None, np.inf
        for d0 in [0.3, 0.5, 0.7]:
            for g0 in [0.3, 0.5, 0.7]:
                result = minimize(objective, x0=[d0, g0],
                                bounds=[(0.01, 2.0), (0.01, 0.99)],
                                method='L-BFGS-B')
                if result.fun < best_mse:
                    best_mse = result.fun
                    best_result = result
        return CountParams(delta=best_result.x[0], gamma=best_result.x[1],
                           beta_scale=0.0), best_mse


# =============================================================================
# AIC COMPUTATION
# =============================================================================

def compute_aic(mse: float, n_obs: int, n_params: int) -> float:
    """Compute AIC from MSE assuming Gaussian errors."""
    sigma = np.sqrt(mse)
    ll = -n_obs/2 * np.log(2 * np.pi) - n_obs * np.log(sigma) - n_obs/2
    return 2 * n_params - 2 * ll


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 1: MODEL FITTING")
    print("=" * 70)

    print("\nLoading data...")
    beliefs, partner_lookup = load_exp1_data()
    samples = get_player_samples(beliefs)

    print("Preparing transitions...")
    slider_trans = prepare_transitions(beliefs, partner_lookup, samples, 'slider')
    chat_trans = prepare_transitions(beliefs, partner_lookup, samples, 'interactive')

    n_slider, n_chat = len(slider_trans), len(chat_trans)
    print(f"  Slider: {n_slider} transitions")
    print(f"  Chat: {n_chat} transitions")

    # =========================================================================
    # FIT ALL MODELS (full and ablated)
    # =========================================================================

    print("\n" + "-" * 70)
    print("SLIDER CONDITION")
    print("-" * 70)

    # DeGroot (no anchor)
    dg0_slider, dg0_slider_mse = fit_degroot(slider_trans, with_anchor=False)
    dg0_slider_aic = compute_aic(dg0_slider_mse, n_slider, 1)
    print(f"  DeGroot:            α={dg0_slider.alpha:.2f}, AIC={dg0_slider_aic:.1f}")

    # DeGroot + Anchoring
    dg_slider, dg_slider_mse = fit_degroot(slider_trans, with_anchor=True)
    dg_slider_aic = compute_aic(dg_slider_mse, n_slider, 2)
    print(f"  DeGroot + Anchor:   α={dg_slider.alpha:.2f}, β={dg_slider.beta:.2f}, AIC={dg_slider_aic:.1f}")

    # Count (no anchor)
    ct0_slider, ct0_slider_mse = fit_count(slider_trans, with_anchor=False)
    ct0_slider_aic = compute_aic(ct0_slider_mse, n_slider, 2)
    print(f"  Count:              δ={ct0_slider.delta:.2f}, γ={ct0_slider.gamma:.2f}, AIC={ct0_slider_aic:.1f}")

    # Count + Relative Anchoring
    ct_slider, ct_slider_mse = fit_count(slider_trans, with_anchor=True)
    ct_slider_aic = compute_aic(ct_slider_mse, n_slider, 3)
    print(f"  Count + RelAnchor:  δ={ct_slider.delta:.2f}, γ={ct_slider.gamma:.2f}, β_s={ct_slider.beta_scale:.2f}, AIC={ct_slider_aic:.1f}")

    print("\n" + "-" * 70)
    print("CHAT CONDITION")
    print("-" * 70)

    # DeGroot (no anchor)
    dg0_chat, dg0_chat_mse = fit_degroot(chat_trans, with_anchor=False)
    dg0_chat_aic = compute_aic(dg0_chat_mse, n_chat, 1)
    print(f"  DeGroot:            α={dg0_chat.alpha:.2f}, AIC={dg0_chat_aic:.1f}")

    # DeGroot + Anchoring
    dg_chat, dg_chat_mse = fit_degroot(chat_trans, with_anchor=True)
    dg_chat_aic = compute_aic(dg_chat_mse, n_chat, 2)
    print(f"  DeGroot + Anchor:   α={dg_chat.alpha:.2f}, β={dg_chat.beta:.2f}, AIC={dg_chat_aic:.1f}")

    # Count (no anchor)
    ct0_chat, ct0_chat_mse = fit_count(chat_trans, with_anchor=False)
    ct0_chat_aic = compute_aic(ct0_chat_mse, n_chat, 2)
    print(f"  Count:              δ={ct0_chat.delta:.2f}, γ={ct0_chat.gamma:.2f}, AIC={ct0_chat_aic:.1f}")

    # Count + Relative Anchoring
    ct_chat, ct_chat_mse = fit_count(chat_trans, with_anchor=True)
    ct_chat_aic = compute_aic(ct_chat_mse, n_chat, 3)
    print(f"  Count + RelAnchor:  δ={ct_chat.delta:.2f}, γ={ct_chat.gamma:.2f}, β_s={ct_chat.beta_scale:.2f}, AIC={ct_chat_aic:.1f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    print("\n" + "=" * 70)
    print("AIC COMPARISON TABLE")
    print("=" * 70)
    print(f"""
                              Slider          Chat           k
                              ------          ----          ---
DeGroot                      {dg0_slider_aic:8.1f}      {dg0_chat_aic:8.1f}          1
DeGroot + Anchoring          {dg_slider_aic:8.1f}      {dg_chat_aic:8.1f}          2
Count                        {ct0_slider_aic:8.1f}      {ct0_chat_aic:8.1f}          2
Count + Relative Anchoring   {ct_slider_aic:8.1f}      {ct_chat_aic:8.1f}          3

Best DeGroot family:         {min(dg0_slider_aic, dg_slider_aic):8.1f}      {min(dg0_chat_aic, dg_chat_aic):8.1f}
Best Count family:           {min(ct0_slider_aic, ct_slider_aic):8.1f}      {min(ct0_chat_aic, ct_chat_aic):8.1f}

ΔAIC (best Count - best DG)  {min(ct0_slider_aic, ct_slider_aic) - min(dg0_slider_aic, dg_slider_aic):+8.1f}      {min(ct0_chat_aic, ct_chat_aic) - min(dg0_chat_aic, dg_chat_aic):+8.1f}

  Positive = DeGroot better, Negative = Count better
""")

    # LaTeX table
    print("=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(f"""
\\begin{{table}}[t]
\\centering
\\caption{{Model comparison (Experiment 1). Lower AIC = better fit. $k$ = number of parameters.}}
\\label{{tab:model-comparison}}
\\begin{{tabular}}{{lccc}}
\\toprule
Model & Slider & Chat & $k$ \\\\
\\midrule
DeGroot & ${dg0_slider_aic:.0f}$ & ${dg0_chat_aic:.0f}$ & 1 \\\\
DeGroot + Anchoring & ${dg_slider_aic:.0f}$ & ${dg_chat_aic:.0f}$ & 2 \\\\
\\midrule
Count & ${ct0_slider_aic:.0f}$ & ${ct0_chat_aic:.0f}$ & 2 \\\\
Count + Relative Anchoring & ${ct_slider_aic:.0f}$ & ${ct_chat_aic:.0f}$ & 3 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

    # Save results
    results = pd.DataFrame([
        {'condition': 'slider', 'model': 'degroot', 'anchor': False,
         'alpha': dg0_slider.alpha, 'mse': dg0_slider_mse, 'aic': dg0_slider_aic, 'k': 1},
        {'condition': 'slider', 'model': 'degroot', 'anchor': True,
         'alpha': dg_slider.alpha, 'beta': dg_slider.beta,
         'mse': dg_slider_mse, 'aic': dg_slider_aic, 'k': 2},
        {'condition': 'slider', 'model': 'count', 'anchor': False,
         'delta': ct0_slider.delta, 'gamma': ct0_slider.gamma,
         'mse': ct0_slider_mse, 'aic': ct0_slider_aic, 'k': 2},
        {'condition': 'slider', 'model': 'count', 'anchor': True,
         'delta': ct_slider.delta, 'gamma': ct_slider.gamma, 'beta_scale': ct_slider.beta_scale,
         'mse': ct_slider_mse, 'aic': ct_slider_aic, 'k': 3},
        {'condition': 'chat', 'model': 'degroot', 'anchor': False,
         'alpha': dg0_chat.alpha, 'mse': dg0_chat_mse, 'aic': dg0_chat_aic, 'k': 1},
        {'condition': 'chat', 'model': 'degroot', 'anchor': True,
         'alpha': dg_chat.alpha, 'beta': dg_chat.beta,
         'mse': dg_chat_mse, 'aic': dg_chat_aic, 'k': 2},
        {'condition': 'chat', 'model': 'count', 'anchor': False,
         'delta': ct0_chat.delta, 'gamma': ct0_chat.gamma,
         'mse': ct0_chat_mse, 'aic': ct0_chat_aic, 'k': 2},
        {'condition': 'chat', 'model': 'count', 'anchor': True,
         'delta': ct_chat.delta, 'gamma': ct_chat.gamma, 'beta_scale': ct_chat.beta_scale,
         'mse': ct_chat_mse, 'aic': ct_chat_aic, 'k': 3},
    ])
    results.to_csv('fit_results.csv', index=False)
    print("\nResults saved to fit_results.csv")
