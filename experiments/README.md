# Experiment code

All experiments share the same [Empirica](https://empirica.ly/) codebase and are run with different `treatment` parameters.

## Experiment 1: Communication channel

Three communication conditions, each with two ground-truth probabilities (0.3 and 0.7):

```
slider-0.3 / slider-0.7
unidirectional-0.3 / unidirectional-0.7
interactive-0.3 / interactive-0.7
```

## Experiment 2: Epistemic difficulty

Interactive chat and slider conditions, crossed with slider direction counterbalancing (RS = Rabbits→Squirrels, SR = Squirrels→Rabbits):

```
RS_interactive / RS_slider
SR_interactive / SR_slider
```

Pre-generated game configurations vary total sample size, representativeness, and information distribution. Game configurations are loaded from JSON files in `server/src/`.
