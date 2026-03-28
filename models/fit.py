"""
Computational Models of Social Learning - Experiment 1 Model Fitting

Fits two model families to Experiment 1 belief update data using TRANSITION-BASED
fitting (predicting belief at t+1 given state at t). This approach directly tests
each model's update rule against human behavior.

Models
------
1. DeGroot (Slider condition)
   - Uses partner's BELIEF (transmitted via slider interface)
   - Update: degroot_step = α × own_belief + (1-α) × partner_belief
   - Response: belief = w_update × degroot_step + w_mle × own_mle + w_mid × 0.5
   - Parameters: α (self-weight), w_update, w_mle, w_mid (response weights)

2. Count (Chat condition)
   - Uses partner's COUNTS (transmitted via chat interface)
   - Pooled MLE: pooled = (own_r + δ×cum_r) / (own_n + δ×cum_n)
   - Update: count_step = α × own_belief + (1-α) × pooled_mle
   - Response: belief = w_update × count_step + w_mle × own_mle + w_mid × 0.5
   - Parameters: δ (evidence discount), α (self-weight), w_update, w_mle, w_mid

Key Theoretical Distinction
---------------------------
The information available differs by condition:
  - Slider: Partners share beliefs → DeGroot averaging is natural
  - Chat: Partners share counts → Evidence pooling is natural

The crossover prediction: DeGroot fits slider better, Count fits chat better.

Output
------
- fit_results.csv: All fitted parameters, MSE, AIC, and σ (derived from √MSE)
- Console: AIC comparison table and LaTeX-formatted table

Usage
-----
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
    """DeGroot with response weights: 3 model parameters + derived sigma."""
    alpha: float     # Self-weight in DeGroot averaging
    w_update: float  # Weight on DeGroot update
    w_mle: float     # Weight on own MLE
    w_mid: float     # Weight on 0.5 (midpoint anchor)
    sigma: float     # Response noise std dev (derived from MSE)


@dataclass
class CountParams:
    """Count with response weights: 4 model parameters + derived sigma."""
    delta: float     # Evidence discount for others' counts
    alpha: float     # Self-weight on current belief
    w_update: float  # Weight on count update
    w_mle: float     # Weight on own MLE
    w_mid: float     # Weight on 0.5 (midpoint anchor)
    sigma: float     # Response noise std dev (derived from MSE)


# =============================================================================
# MODEL PREDICTIONS
# =============================================================================

def degroot_predict(row, params: DeGrootParams) -> float:
    """
    DeGroot prediction with linear response combination.

    Update rule:
        degroot_step = α × own_belief + (1-α) × partner_belief

    Response:
        belief = w_update × degroot_step + w_mle × own_mle + w_mid × 0.5
    """
    degroot_step = params.alpha * row['own_belief'] + (1 - params.alpha) * row['partner_belief']
    belief = params.w_update * degroot_step + params.w_mle * row['own_mle'] + params.w_mid * 0.5
    return belief


def count_predict(row, params: CountParams) -> float:
    """
    Count prediction with linear response combination.

    Update rule:
        pooled_mle = (own_r + δ × cumulative_r) / (own_n + δ × cumulative_n)
        count_step = α × own_belief + (1-α) × pooled_mle

    Response:
        belief = w_update × count_step + w_mle × own_mle + w_mid × 0.5
    """
    own_n = row['own_r'] + row['own_s']
    cumulative_n = row['cumulative_r'] + row['cumulative_s']

    # Pooled MLE with discounted evidence
    pooled_r = row['own_r'] + params.delta * row['cumulative_r']
    pooled_n = own_n + params.delta * cumulative_n
    pooled_mle = pooled_r / pooled_n if pooled_n > 0 else 0.5

    # Self-weight step
    count_step = params.alpha * row['own_belief'] + (1 - params.alpha) * pooled_mle

    belief = params.w_update * count_step + params.w_mle * row['own_mle'] + params.w_mid * 0.5
    return belief


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_degroot(transitions: pd.DataFrame, with_anchor: bool = True) -> Tuple[DeGrootParams, float]:
    """Fit DeGroot model. Returns params and MSE."""
    actual = transitions['actual_next'].values

    if with_anchor:
        # Full model: α + response weights (w_update, w_mle, w_mid)
        # Weights are free parameters (not constrained to sum to 1)
        def objective(params_array):
            alpha, w_update, w_mle, w_mid = params_array
            if not (0 < alpha < 1):
                return 1e10
            params = DeGrootParams(alpha=alpha, w_update=w_update, w_mle=w_mle,
                                  w_mid=w_mid, sigma=0.0)
            preds = transitions.apply(lambda r: degroot_predict(r, params), axis=1).values
            return np.mean((preds - actual) ** 2)

        best_result, best_mse = None, np.inf
        for a0 in [0.6, 0.75, 0.85]:
            for wu0 in [0.5, 0.7, 0.9]:
                for wm0 in [0.0, 0.1, 0.2]:
                    for wr0 in [0.0, 0.1, 0.3]:
                        result = minimize(objective, x0=[a0, wu0, wm0, wr0],
                                        bounds=[(0.01, 0.99), (-2.0, 2.0), (-1.0, 1.0), (-1.0, 2.0)],
                                        method='L-BFGS-B')
                        if result.fun < best_mse:
                            best_mse = result.fun
                            best_result = result

        sigma = np.sqrt(best_mse)
        return DeGrootParams(alpha=best_result.x[0], w_update=best_result.x[1],
                            w_mle=best_result.x[2], w_mid=best_result.x[3],
                            sigma=sigma), best_mse

    else:
        # Base model: α only (1 param), w_update=1.0, w_mle=0.0, w_mid=0.0
        def objective(params_array):
            alpha = params_array[0]
            if not (0 < alpha < 1):
                return 1e10
            params = DeGrootParams(alpha=alpha, w_update=1.0, w_mle=0.0, w_mid=0.0, sigma=0.0)
            preds = transitions.apply(lambda r: degroot_predict(r, params), axis=1).values
            return np.mean((preds - actual) ** 2)

        best_result, best_mse = None, np.inf
        for a0 in [0.6, 0.75, 0.85]:
            result = minimize(objective, x0=[a0],
                            bounds=[(0.01, 0.99)],
                            method='L-BFGS-B')
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result

        sigma = np.sqrt(best_mse)
        return DeGrootParams(alpha=best_result.x[0], w_update=1.0, w_mle=0.0,
                            w_mid=0.0, sigma=sigma), best_mse


def fit_count(transitions: pd.DataFrame, with_anchor: bool = True) -> Tuple[CountParams, float]:
    """Fit Count model. Returns params and MSE."""
    actual = transitions['actual_next'].values

    if with_anchor:
        # Full model: δ, α + response weights (free parameters)
        def objective(params_array):
            delta, alpha, w_update, w_mle, w_mid = params_array
            if not (0 < delta < 3 and 0 < alpha < 1):
                return 1e10
            params = CountParams(delta=delta, alpha=alpha, w_update=w_update,
                               w_mle=w_mle, w_mid=w_mid, sigma=0.0)
            preds = transitions.apply(lambda r: count_predict(r, params), axis=1).values
            return np.mean((preds - actual) ** 2)

        best_result, best_mse = None, np.inf
        for d0 in [0.3, 0.5, 0.7]:
            for a0 in [0.3, 0.5, 0.7]:
                for wu0 in [0.5, 0.7, 0.9]:
                    for wm0 in [0.0, 0.1, 0.2]:
                        for wr0 in [0.0, 0.1, 0.3]:
                            result = minimize(objective, x0=[d0, a0, wu0, wm0, wr0],
                                            bounds=[(0.01, 2.0), (0.01, 0.99),
                                                    (-2.0, 2.0), (-1.0, 1.0), (-1.0, 2.0)],
                                            method='L-BFGS-B')
                            if result.fun < best_mse:
                                best_mse = result.fun
                                best_result = result

        sigma = np.sqrt(best_mse)
        return CountParams(delta=best_result.x[0], alpha=best_result.x[1],
                          w_update=best_result.x[2], w_mle=best_result.x[3],
                          w_mid=best_result.x[4], sigma=sigma), best_mse

    else:
        # Base model: δ, α only (2 params), w_update=1.0, w_mle=0.0, w_mid=0.0
        def objective(params_array):
            delta, alpha = params_array
            if not (0 < delta < 3 and 0 < alpha < 1):
                return 1e10
            params = CountParams(delta=delta, alpha=alpha, w_update=1.0,
                               w_mle=0.0, w_mid=0.0, sigma=0.0)
            preds = transitions.apply(lambda r: count_predict(r, params), axis=1).values
            return np.mean((preds - actual) ** 2)

        best_result, best_mse = None, np.inf
        for d0 in [0.3, 0.5, 0.7]:
            for a0 in [0.3, 0.5, 0.7]:
                result = minimize(objective, x0=[d0, a0],
                                bounds=[(0.01, 2.0), (0.01, 0.99)],
                                method='L-BFGS-B')
                if result.fun < best_mse:
                    best_mse = result.fun
                    best_result = result

        sigma = np.sqrt(best_mse)
        return CountParams(delta=best_result.x[0], alpha=best_result.x[1],
                          w_update=1.0, w_mle=0.0, w_mid=0.0, sigma=sigma), best_mse


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
    # FIT ALL MODELS (with and without anchoring)
    # =========================================================================

    fits = {}

    for cond_name, trans, n_obs in [('slider', slider_trans, n_slider), ('chat', chat_trans, n_chat)]:
        print(f"\n{'-'*70}\n{cond_name.upper()} CONDITION\n{'-'*70}")

        # DeGroot family
        for anchor in [False, True]:
            p, mse = fit_degroot(trans, with_anchor=anchor)
            k = 1 + 3*int(anchor)  # 1 param base, +3 for anchor (w_update, w_mle, w_mid)
            aic = compute_aic(mse, n_obs, k)

            label = "DeGroot" + (" + Response" if anchor else "")
            fits[(cond_name, 'degroot', anchor)] = {'params': p, 'mse': mse, 'aic': aic, 'k': k}
            if anchor:
                print(f"  {label:25s} α={p.alpha:.2f}, wu={p.w_update:.2f}, wm={p.w_mle:.2f}, wr={p.w_mid:.2f}, σ={p.sigma:.3f}, k={k}, AIC={aic:.1f}")
            else:
                print(f"  {label:25s} α={p.alpha:.2f}, σ={p.sigma:.3f}, k={k}, AIC={aic:.1f}")

        # Count family
        for anchor in [False, True]:
            p, mse = fit_count(trans, with_anchor=anchor)
            k = 2 + 3*int(anchor)  # 2 params base, +3 for anchor (w_update, w_mle, w_mid)
            aic = compute_aic(mse, n_obs, k)

            label = "Count" + (" + Response" if anchor else "")
            fits[(cond_name, 'count', anchor)] = {'params': p, 'mse': mse, 'aic': aic, 'k': k}
            if anchor:
                print(f"  {label:25s} δ={p.delta:.2f}, α={p.alpha:.2f}, wu={p.w_update:.2f}, wm={p.w_mle:.2f}, wr={p.w_mid:.2f}, σ={p.sigma:.3f}, k={k}, AIC={aic:.1f}")
            else:
                print(f"  {label:25s} δ={p.delta:.2f}, α={p.alpha:.2f}, σ={p.sigma:.3f}, k={k}, AIC={aic:.1f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    print("\n" + "=" * 70)
    print("AIC COMPARISON TABLE")
    print("=" * 70)

    models = [
        ("DeGroot", 'degroot', False),
        ("DeGroot + Response", 'degroot', True),
        ("Count", 'count', False),
        ("Count + Response", 'count', True),
    ]

    print(f"\n{'Model':<30s} {'Slider':>10s} {'Chat':>10s} {'k':>5s}")
    print("-" * 60)
    for label, model, anchor in models:
        s_aic = fits[('slider', model, anchor)]['aic']
        c_aic = fits[('chat', model, anchor)]['aic']
        k = fits[('slider', model, anchor)]['k']
        print(f"{label:<30s} {s_aic:10.1f} {c_aic:10.1f} {k:5d}")

    # Best in each family (with anchoring)
    best_dg_slider = fits[('slider', 'degroot', True)]['aic']
    best_dg_chat = fits[('chat', 'degroot', True)]['aic']
    best_ct_slider = fits[('slider', 'count', True)]['aic']
    best_ct_chat = fits[('chat', 'count', True)]['aic']

    print("-" * 60)
    print(f"\n{'ΔAIC (Count - DeGroot)':<30s} {best_ct_slider - best_dg_slider:+10.1f} {best_ct_chat - best_dg_chat:+10.1f}")
    print("  (comparing full models with anchoring)")
    print("  Positive = DeGroot better, Negative = Count better")

    # LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print("""
\\begin{table}[t]
\\centering
\\caption{Model comparison (Experiment 1). Lower AIC = better fit. $k$ = number of parameters.}
\\label{tab:model-comparison}
\\begin{tabular}{lccc}
\\toprule
Model & Slider & Chat & $k$ \\\\
\\midrule""")
    for label, model, anchor in models:
        s_aic = fits[('slider', model, anchor)]['aic']
        c_aic = fits[('chat', model, anchor)]['aic']
        k = fits[('slider', model, anchor)]['k']
        if label == "Count":
            print("\\midrule")
        print(f"{label} & ${s_aic:.0f}$ & ${c_aic:.0f}$ & {k} \\\\")
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    # Save results
    rows = []
    for (cond, model, anchor), data in fits.items():
        p = data['params']
        row = {
            'condition': cond,
            'model': model,
            'anchor': anchor,
            'mse': data['mse'],
            'aic': data['aic'],
            'k': data['k'],
            'sigma': p.sigma,
            'w_update': p.w_update,
            'w_mle': p.w_mle,
            'w_mid': p.w_mid,
        }
        if model == 'degroot':
            row.update({'alpha': p.alpha})
        else:
            row.update({'alpha': p.alpha, 'delta': p.delta})
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv('fit_results.csv', index=False)
    print("\nResults saved to fit_results.csv")

    # Print best params for manuscript
    print("\n" + "=" * 70)
    print("BEST PARAMETERS FOR MANUSCRIPT")
    print("=" * 70)
    dg_full = fits[('slider', 'degroot', True)]['params']
    ct_full = fits[('chat', 'count', True)]['params']
    print(f"Slider (DeGroot+Response): α={dg_full.alpha:.2f}, wu={dg_full.w_update:.2f}, wm={dg_full.w_mle:.2f}, wr={dg_full.w_mid:.2f}, σ={dg_full.sigma:.3f}")
    print(f"Chat (Count+Response):     δ={ct_full.delta:.2f}, α={ct_full.alpha:.2f}, wu={ct_full.w_update:.2f}, wm={ct_full.w_mle:.2f}, wr={ct_full.w_mid:.2f}, σ={ct_full.sigma:.3f}")
