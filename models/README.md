# Computational Models

This directory contains scripts for fitting and evaluating computational models of belief updating in social learning experiments.

## Model Fitting

**`fit.py`** - Single entry point for fitting all models to Experiment 1 data.

```bash
python fit.py
```

### Models

Two model families with nested variants:

1. **DeGroot family** (belief averaging)
   - DeGroot: 1 parameter (α = self-weight)
   - DeGroot + Anchoring: 2 parameters (α, β = anchor to own MLE)

2. **Count family** (evidence pooling)
   - Count: 2 parameters (δ = evidence discount, γ = inertia)
   - Count + Relative Anchoring: 3 parameters (δ, γ, β_scale)

The key theoretical distinction:
- **DeGroot** uses partner's *belief* (available in slider condition)
- **Count** uses partner's *counts* (available in chat condition)

### Fitting Approach

**Transition-based**: Fits round-to-round belief transitions, directly testing each model's update rule.

### Key Results (Experiment 1)

| Model | Slider | Chat | k |
|-------|--------|------|---|
| DeGroot | -2713 | -2574 | 1 |
| DeGroot + Anchoring | **-2804** | -2769 | 2 |
| Count | -2511 | -2901 | 2 |
| Count + Relative Anchoring | -2519 | **-2910** | 3 |

**Crossover confirmed**: DeGroot wins slider (+284 ΔAIC), Count wins chat (-141 ΔAIC)

## Other Files

- **`simulate_exp2_predictions.py`** - Generate model predictions for Experiment 2
- **`make_model_fig5.py`** - Create model predictions figure
- **`plot_models.py`** - Visualization utilities
