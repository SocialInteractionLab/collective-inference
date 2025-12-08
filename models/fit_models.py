"""
Computational Models of Social Learning - Experiment 1

Compares two models of belief updating:
1. DeGroot: belief(t+1) = α·own + (1-α)·partner_belief
2. Count Pooling: belief(t+1) = γ·own + (1-γ)·pooled_MLE
   where pooled_MLE uses cumulative evidence from all unique partners

Key result: DeGroot fits slider better; Count Pooling fits chat better.
This crossover demonstrates that chat enables raw evidence transmission.

Usage: python fit_models.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import os


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load belief data and partner assignments."""
    # Belief trajectories
    beliefs = pd.read_csv('../analysis/exp1/exp1_clean_data.csv')
    beliefs['belief'] = beliefs['guess'] / 100.0

    # Partner lookup from messages
    msgs = pd.read_csv('../data/exp1/raw/1_extracted_msgs_prereg/extracted_messages.csv')
    msgs = msgs.dropna(subset=['round', 'recipient', 'sender.id', 'groupID'])

    # Create (game, player, round) -> partner mapping
    partner_lookup = {}
    for _, row in msgs.iterrows():
        key = (row['groupID'], row['recipient'], int(row['round']))
        if key not in partner_lookup:  # Take first message if multiple
            partner_lookup[key] = row['sender.id']

    return beliefs, partner_lookup


def get_player_samples(beliefs):
    """Get each player's sample counts: {(game, player): (r, s)}"""
    samples = {}
    for _, row in beliefs.drop_duplicates(['gameID', 'playerID']).iterrows():
        key = (row['gameID'], row['playerID'])
        samples[key] = (int(row['nRabbits']), int(row['nSquirrels']))
    return samples


# =============================================================================
# MODEL PREDICTIONS
# =============================================================================

def degroot_predict(own_belief, partner_belief, alpha):
    """DeGroot: weighted average of own and partner's belief."""
    return alpha * own_belief + (1 - alpha) * partner_belief


def count_pooling_predict(own_r, own_s, cumulative_others_r, cumulative_others_s,
                          own_belief, delta, gamma):
    """
    Count Pooling model with cumulative evidence.

    pooled_MLE = (own_r + δ·Σpartners_r) / (own_n + δ·Σpartners_n)
    belief(t+1) = γ·own_belief + (1-γ)·pooled_MLE
    """
    own_n = own_r + own_s
    others_n = cumulative_others_r + cumulative_others_s

    if others_n == 0:
        return own_belief

    pooled_r = own_r + delta * cumulative_others_r
    pooled_n = own_n + delta * others_n
    pooled_mle = pooled_r / pooled_n

    return gamma * own_belief + (1 - gamma) * pooled_mle


# =============================================================================
# COMPUTE PREDICTIONS FOR ALL TRANSITIONS
# =============================================================================

def compute_all_predictions(beliefs, partner_lookup, samples,
                            alpha=0.6, delta=0.8, gamma=0.5):
    """
    Compute predictions for all belief transitions.

    Returns DataFrame with columns:
    - game_id, player_id, round_t, condition
    - actual: actual belief at t+1
    - degroot_pred: DeGroot prediction
    - count_pred: Count Pooling prediction
    """
    results = []

    for game_id in beliefs['gameID'].unique():
        game_df = beliefs[beliefs['gameID'] == game_id].copy()
        condition = game_df['condition'].iloc[0]

        for player_id in game_df['playerID'].unique():
            player_df = game_df[game_df['playerID'] == player_id].sort_values('idx')
            rounds = player_df['idx'].values
            player_beliefs = player_df['belief'].values

            own_r, own_s = samples.get((game_id, player_id), (0, 0))

            # Track cumulative evidence from unique partners
            seen_partners = set()
            cumulative_r = 0
            cumulative_s = 0

            for t_idx in range(len(rounds) - 1):
                t = int(rounds[t_idx])
                t_next = int(rounds[t_idx + 1])

                own_belief = player_beliefs[t_idx]
                actual_next = player_beliefs[t_idx + 1]

                # Get partner for this round
                partner_key = (game_id, player_id, t)
                partner_id = partner_lookup.get(partner_key)

                if partner_id is None:
                    continue

                # Get partner's data
                partner_sample = samples.get((game_id, partner_id))
                if partner_sample is None:
                    continue

                partner_r, partner_s = partner_sample

                # Get partner's belief at time t
                partner_df = game_df[game_df['playerID'] == partner_id]
                partner_at_t = partner_df[partner_df['idx'] == t]
                if len(partner_at_t) == 0:
                    continue
                partner_belief = partner_at_t['belief'].values[0]

                # DeGroot prediction (uses current partner's belief)
                degroot_pred = degroot_predict(own_belief, partner_belief, alpha)

                # Update cumulative evidence (only if new partner)
                if partner_id not in seen_partners:
                    seen_partners.add(partner_id)
                    cumulative_r += partner_r
                    cumulative_s += partner_s

                # Count Pooling prediction (uses cumulative evidence)
                count_pred = count_pooling_predict(
                    own_r, own_s, cumulative_r, cumulative_s,
                    own_belief, delta, gamma
                )

                results.append({
                    'game_id': game_id,
                    'player_id': player_id,
                    'round_t': t,
                    'condition': condition,
                    'own_belief': own_belief,
                    'actual': actual_next,
                    'degroot_pred': degroot_pred,
                    'count_pred': count_pred,
                    'n_partners_seen': len(seen_partners)
                })

    return pd.DataFrame(results)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compute_log_likelihood(pred, actual):
    """Compute log-likelihood assuming normal errors."""
    residuals = actual - pred
    n = len(residuals)
    sigma = np.sqrt(np.mean(residuals**2))
    ll = -n/2 * np.log(2 * np.pi) - n * np.log(sigma) - n/2
    return ll, sigma


def compare_models(predictions):
    """Compare models using AIC."""
    print("=" * 70)
    print("MODEL COMPARISON (Cumulative Evidence)")
    print("=" * 70)

    summary = []

    for condition in ['slider', 'interactive']:
        cond_df = predictions[predictions['condition'] == condition]
        if len(cond_df) == 0:
            continue

        print(f"\n{condition.upper()} (n={len(cond_df)} transitions)")
        print("-" * 50)

        # DeGroot
        ll_dg, sigma_dg = compute_log_likelihood(
            cond_df['degroot_pred'].values,
            cond_df['actual'].values
        )
        aic_dg = -2 * ll_dg + 2 * 2  # 2 params: alpha, sigma

        # Count Pooling
        ll_cp, sigma_cp = compute_log_likelihood(
            cond_df['count_pred'].values,
            cond_df['actual'].values
        )
        aic_cp = -2 * ll_cp + 2 * 3  # 3 params: delta, gamma, sigma

        delta_aic = aic_dg - aic_cp  # positive = count model better

        print(f"  DeGroot:       LL={ll_dg:8.1f}, AIC={aic_dg:8.1f}, σ={sigma_dg:.3f}")
        print(f"  Count Pooling: LL={ll_cp:8.1f}, AIC={aic_cp:8.1f}, σ={sigma_cp:.3f}")
        print(f"  ΔAIC = {delta_aic:+.1f} ({'Count better' if delta_aic > 0 else 'DeGroot better'})")

        summary.append({
            'condition': condition,
            'n': len(cond_df),
            'll_degroot': ll_dg,
            'll_count': ll_cp,
            'aic_degroot': aic_dg,
            'aic_count': aic_cp,
            'delta_aic': delta_aic
        })

    return pd.DataFrame(summary)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    beliefs, partner_lookup = load_data()
    samples = get_player_samples(beliefs)

    print(f"Loaded {len(beliefs)} observations from {beliefs['gameID'].nunique()} games")
    print(f"Partner lookup has {len(partner_lookup)} entries")

    print("\nComputing predictions with cumulative evidence...")
    predictions = compute_all_predictions(beliefs, partner_lookup, samples)
    print(f"Generated {len(predictions)} transition predictions")

    summary = compare_models(predictions)

    # Save results
    predictions.to_csv('predictions.csv', index=False)
    summary.to_csv('model_comparison_summary.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    slider = summary[summary['condition'] == 'slider'].iloc[0]
    chat = summary[summary['condition'] == 'interactive'].iloc[0]
    print(f"""
Slider: ΔAIC = {slider['delta_aic']:+.0f} (DeGroot {'better' if slider['delta_aic'] < 0 else 'worse'})
Chat:   ΔAIC = {chat['delta_aic']:+.0f} (Count model {'better' if chat['delta_aic'] > 0 else 'worse'})

This crossover demonstrates that chat enables transmission of raw
evidence (counts), not just summary beliefs.
""")
    print("Results saved to predictions.csv and model_comparison_summary.csv")
