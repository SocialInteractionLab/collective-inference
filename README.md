# Collective inference depends on the communication channel

This repository contains code, data, and analyses for our manuscript "Collective inference depends on the communication channel."

## Repository Structure

### `/experiments`
[Empirica](https://empirica.ly/) experiment code for the web-based platform (client and server).

### `/data`
Experimental data organized by experiment (exp1, exp2) with raw Empirica exports and merged datasets.

### `/analysis`
Quarto notebooks reproducing all reported statistics:
- `exp1/exp1_main_analysis.qmd` — Experiment 1 analyses
- `exp2/1_data_cleaning.qmd` — Experiment 2 data preparation
- `exp2/exp2_main_analysis.qmd` — Experiment 2 analyses

### `/models`
Computational models:
- `fit.py` — Fits Belief (DeGroot) and Evidence (count-based) models to Experiment 1 data
- `predict.py` — Generates Experiment 2 predictions from fitted parameters

### `/stimuli`
Game generation scripts for Experiment 2 stimuli. See `stimuli/README.md` for details.
