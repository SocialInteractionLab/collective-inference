# Computational Models

This directory contains scripts for fitting and evaluating computational models of belief updating in social learning experiments.

## Scripts

### `fit.py` - Model Fitting (Experiment 1)

Fits two model families to Experiment 1 belief update data using transition-based fitting.

```bash
python fit.py
```

**Models:**

1. **DeGroot** (Slider condition)
   - Uses partner's BELIEF (transmitted via slider)
   - Update: `degroot = α × own_belief + (1-α) × partner_belief`
   - Response: `belief = w_update × degroot + w_mle × own_mle + w_mid × 0.5`
   - Parameters: α, w_update, w_mle, w_mid

2. **Count** (Chat condition)
   - Uses partner's COUNTS (transmitted via chat)
   - Update: `pooled_mle = (own_r + δ×cum_r) / (own_n + δ×cum_n)`
   - Inertia: `count_step = γ × own_belief + (1-γ) × pooled_mle`
   - Response: `belief = w_update × count_step + w_mle × own_mle + w_mid × 0.5`
   - Parameters: δ, γ, w_update, w_mle, w_mid

**Output:** `fit_results.csv` with all parameters, AIC, MSE, and σ (derived from √MSE)

### `predict.py` - Model Predictions (Experiment 2)

Uses fitted parameters to simulate Experiment 2 and generate predictions across difficulty axes.

```bash
python predict.py
```

**Output:** `figures/model_fig5.pdf` - Predicted error across difficulty axes (compare to `figures/fig5.pdf` for human data)

## Key Results

### Experiment 1: Model Comparison (AIC)

| Model | Slider | Chat | k |
|-------|--------|------|---|
| DeGroot | -2713 | -2574 | 1 |
| DeGroot + Response | **-2801** | -2795 | 4 |
| Count | -2511 | -2901 | 2 |
| Count + Response | -2541 | **-2917** | 5 |

**Crossover confirmed:** DeGroot fits slider better, Count fits chat better.

### Experiment 2: Predictions

The models correctly predict:
- Chat condition has lower error than Slider (Count pools evidence efficiently)
- Representativeness has largest effect (both conditions)
- Distribution evenness reduces error (both conditions)
