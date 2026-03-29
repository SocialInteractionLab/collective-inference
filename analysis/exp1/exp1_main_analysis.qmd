---
title: "Experiment 1: Main Analysis (Individual-Level Exclusions)"
format: html
editor: visual
---

## Import packages

```{r, message=FALSE, warning=FALSE}
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
library(lmerTest)
conflicts_prefer(lmerTest::lmer)
library(tidyverse)
library(tidyboot)
library(lme4)
library(here)
library(emmeans)
library(ggthemes)
library(patchwork)

theme_set(theme_few())
```

## Load and prepare data

This analysis uses the merged data file created by `data_preparation.qmd`, which documents:
- N=973 participants recruited from Prolific
- N=890 participants provided ≥1 response (in this file)
- N=83 participants dropped out (8.5% attrition)

```{r, message=FALSE, warning=FALSE}
# Load merged data file (see data_preparation.qmd for complete pipeline)
d.raw <- read_csv(here('data/exp1/PreregMergedGuesses.csv'))

cat("Loaded PreregMergedGuesses.csv:\n")
cat("  Total games:", n_distinct(d.raw$gameID), "\n")
cat("  Games with valid treatments:", n_distinct(d.raw$gameID[!is.na(d.raw$treatmentName)]), "\n")
cat("  Participants:", n_distinct(d.raw$playerID), "\n")

# Filter to valid treatments for analysis
d.guesses <- d.raw %>%
  filter(!is.na(treatmentName))

# Handle duplicate rows in unidirectional condition
unique_rows <- d.guesses %>%
  filter(treatmentName %in% c("rerun-unidirectional- 0.7", "rerun-unidirectional- 0.3")) %>%
  distinct(playerID, gameID, guess, roundID, .keep_all = TRUE)

d.guesses <- d.guesses %>%
  filter(!(treatmentName %in% c("rerun-unidirectional- 0.7", "rerun-unidirectional- 0.3"))) %>%
  bind_rows(unique_rows)

cat("After cleaning duplicate rows:\n")
cat("  Participants:", n_distinct(d.guesses$playerID), "participants in",
    n_distinct(d.guesses$gameID), "games\n")
```

## Apply individual-level exclusions (Preregistered)

Following our preregistration, exclude participants who failed to respond on >20% of trials (i.e., <80% response rate). Keep all networks regardless of size (intent-to-treat at network level).

```{r}
# Calculate response rate per participant (preregistered exclusion criterion)
participant_responses <- d.guesses %>%
  group_by(playerID) %>%
  summarise(
    total_rounds = n_distinct(idx),
    response_rate = total_rounds / 12,
    .groups = "drop"
  )

# Exclude participants with <80% response rate (i.e., >20% missing)
exclude_prereg <- participant_responses %>%
  filter(response_rate < 0.80) %>%
  pull(playerID)

cat("Excluded", length(exclude_prereg), "participants with <80% response rate\n")

# Remove only inattentive individuals (keep all networks)
d.guesses <- d.guesses %>%
  filter(!(playerID %in% exclude_prereg))

cat("Final sample:", n_distinct(d.guesses$playerID), "participants in",
    n_distinct(d.guesses$gameID), "networks\n")
```

## Clean and prepare data

Data file already contains game-level statistics (nRabbitsGame, nSquirrelsGame, mleEstimateGame) computed during data preparation.

```{r}
# Clean treatment names and compute error
d.guesses <- d.guesses %>%
  mutate(
    # Clean treatment names
    treatmentName = case_when(
      treatmentName == "rerun-unidirectional- 0.7" ~ 'unidirectional-0.7',
      treatmentName == "rerun-unidirectional- 0.3" ~ 'unidirectional-0.3',
      treatmentName == "rerun-interactive-0.3" ~ 'interactive-0.3',
      treatmentName == "rerun-interactive- 0.7" ~ 'interactive-0.7',
      treatmentName == "rerun-slider- 0.3" ~ 'slider-0.3',
      treatmentName == "rerun-slider- 0.7" ~ 'slider-0.7',
      TRUE ~ treatmentName
    )
  ) %>%
  separate(treatmentName, into = c('condition', 'treatment'), sep = '-') %>%
  mutate(
    # Calculate error (mleEstimateGame already in data)
    error = abs(guess - mleEstimateGame),
    condition = factor(condition, levels = c("slider", "unidirectional", "interactive"))
  ) %>%
  arrange(playerID, idx)

# Report sample sizes by condition
sample_summary <- d.guesses %>%
  filter(idx == 1) %>%
  group_by(condition, treatment) %>%
  summarise(
    n_networks = n_distinct(gameID),
    n_participants = n_distinct(playerID),
    .groups = "drop"
  )

print(sample_summary)

# Create shifted data for proper baseline alignment
# Slider round 1 = pre-social baseline (confirmed via Chat.jsx)
# Shift slider to idx 0-11, others remain 1-12
d.shifted <- d.guesses %>%
  mutate(
    idx_shifted = if_else(condition == "slider", idx - 1, idx),
    idx_final_centered = if_else(condition == "slider",
                                   idx_shifted - 11,  # Slider final = round 11
                                   idx_shifted - 12)  # Others final = round 12
  )

cat("\nShifted data ranges:\n")
cat("Slider: idx", min(d.shifted$idx_shifted[d.shifted$condition == "slider"]), "to",
    max(d.shifted$idx_shifted[d.shifted$condition == "slider"]), "\n")
cat("Other conditions: idx", min(d.shifted$idx_shifted[d.shifted$condition != "slider"]), "to",
    max(d.shifted$idx_shifted[d.shifted$condition != "slider"]), "\n")
```

## Analysis 1: Communication improves inferential accuracy

Test whether groups converge toward the true probability over time using raw centered polynomial terms.

```{r}
# Center the round index for interpretable coefficients
d.guesses <- d.guesses %>%
  mutate(idx_c = idx - mean(idx))

# Model: Does error decrease over time (linear + quadratic)?
# Raw centered polynomial: idx_c captures linear decrease, I(idx_c^2) captures deceleration
mod_convergence <- lmer(
  error ~ idx_c + I(idx_c^2) + (1 + idx_c | playerID),
  data = d.guesses,
  control = lmerControl(optimizer = "bobyqa")
)

summary(mod_convergence)
```

## Analysis 2: Error decreases over time

```{r}
# Prepare data with empirical baseline (round 0)
# Slider round 1 = pre-social baseline (used for all conditions)
slider_data <- d.guesses %>%
  filter(condition == "slider") %>%
  mutate(idx = idx - 1) %>%
  group_by(condition, idx) %>%
  tidyboot_mean(error, na.rm = TRUE)

slider_baseline <- slider_data %>%
  filter(idx == 0) %>%
  select(empirical_stat, ci_lower, ci_upper) %>%
  summarise(empirical_stat = mean(empirical_stat),
            ci_lower = mean(ci_lower),
            ci_upper = mean(ci_upper)) %>%
  mutate(idx = 0)

other_conditions_round0 <- tibble(
  condition = factor(c("interactive", "unidirectional"),
                     levels = c("slider", "unidirectional", "interactive")),
  idx = 0,
  empirical_stat = slider_baseline$empirical_stat,
  ci_lower = slider_baseline$ci_lower,
  ci_upper = slider_baseline$ci_upper
)

other_conditions_actual <- d.guesses %>%
  filter(condition != "slider") %>%
  group_by(condition, idx) %>%
  tidyboot_mean(error, na.rm = TRUE)

error_with_baseline <- bind_rows(slider_data, other_conditions_round0, other_conditions_actual) %>%
  mutate(show_ribbon = (condition == "slider") | (idx > 0),
         show_smooth = (condition == "slider") | (idx > 0))

# Panel A: Error by condition
p_error <- ggplot(error_with_baseline, aes(x = idx, y = empirical_stat, color = condition, fill = condition)) +
    geom_ribbon(data = filter(error_with_baseline, show_ribbon),
                aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.15, color = NA) +
    geom_point(size = 2) +
    geom_line(data = filter(error_with_baseline, idx <= 1, condition != "slider"),
              linetype = "dashed", linewidth = 0.8) +
    geom_smooth(data = filter(error_with_baseline, show_smooth),
                method = 'lm', formula = y ~ poly(x, 2), se = FALSE, linewidth = 1) +
    scale_color_manual(
      name = NULL,
      values = c("slider" = "#E69F00", "unidirectional" = "#56B4E9", "interactive" = "#009E73"),
      labels = c("Slider", "Unidirectional", "Interactive"),
      breaks = c("slider", "unidirectional", "interactive"),
      drop = FALSE
    ) +
    scale_fill_manual(
      name = NULL,
      values = c("slider" = "#E69F00", "unidirectional" = "#56B4E9", "interactive" = "#009E73"),
      labels = c("Slider", "Unidirectional", "Interactive"),
      breaks = c("slider", "unidirectional", "interactive"),
      drop = FALSE
    ) +
    scale_x_continuous(breaks = c(1, 4, 7, 10)) +
    labs(
      x = "Round",
      y = "Mean Absolute Error",
      tag = "A"
    ) +
    theme_few(base_size = 12) +
    theme(
      legend.position = c(0.85, 0.85),
      legend.title = element_blank(),
      legend.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(face = "bold", size = 18),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14)
    )

# Model: Does error decrease over time?
# Note: Using (1 + idx | playerID) - random slopes at participant level
mod_error <- lmer(
  error ~ idx * condition + (1 + idx | playerID),
  data = d.guesses,
  control = lmerControl(optimizer = "bobyqa")
)

summary(mod_error)
```

## Analysis 3: Condition differences at final round

Test whether communication modality affects accuracy at convergence (final round).

```{r}
# Relevel to use interactive as baseline for comparisons
d.shifted_interactive <- d.shifted %>%
  mutate(condition_interactive = relevel(condition, ref = "interactive"))

# Test linear vs polynomial models
mod_condition_linear <- lmer(
  error ~ idx_final_centered * condition_interactive + (1 + idx_final_centered | playerID),
  data = d.shifted_interactive,
  control = lmerControl(optimizer = "bobyqa")
)

mod_condition_poly <- lmer(
  error ~ poly(idx_final_centered, 2) * condition_interactive + (1 + idx_final_centered | playerID),
  data = d.shifted_interactive,
  control = lmerControl(optimizer = "bobyqa")
)

# Likelihood ratio test
cat("Likelihood ratio test (linear vs polynomial):\n")
anova(mod_condition_linear, mod_condition_poly)

cat("\n\nPolynomial model summary:\n")
summary(mod_condition_poly)

cat("\n\nFinal round means by condition:\n")
d.shifted %>%
  filter((condition == "slider" & idx_shifted == 11) |
         (condition != "slider" & idx_shifted == 12)) %>%
  group_by(condition) %>%
  summarise(
    mean_error = mean(error, na.rm = TRUE),
    se_error = sd(error, na.rm = TRUE) / sqrt(n()),
    n = n()
  )
```

## Analysis 4: Adaptive weighting based on participant-observable cues

```{r}
# Calculate participant-observable measures
d.participant_info <- d.guesses %>%
  group_by(playerID, gameID) %>%
  mutate(
    sample_size = nRabbits + nSquirrels  # Observable to participant
  ) %>%
  ungroup()

# Calculate social discrepancy (difference from others' estimates at each round)
d.social_context <- d.participant_info %>%
  group_by(gameID, idx) %>%
  mutate(
    others_mean_estimate = (sum(guess) - guess) / (n() - 1),  # Exclude self
    social_discrepancy = abs(guess - others_mean_estimate)
  ) %>%
  ungroup()

# Calculate belief revisions
d.revisions <- d.social_context %>%
  arrange(playerID, idx) %>%
  group_by(playerID) %>%
  mutate(
    belief_revision = abs(guess - lag(guess)),
    prior_social_discrepancy = lag(social_discrepancy)
  ) %>%
  ungroup() %>%
  filter(!is.na(belief_revision), !is.na(prior_social_discrepancy))

# Model: Adaptive weighting based on sample size and social discrepancy
# Include idx as main effect to control for decreasing revision over time,
# but focus on the sample_size × discrepancy interaction
mod_adaptive <- lmer(
  belief_revision ~ scale(sample_size) * scale(prior_social_discrepancy) + scale(idx) +
    (1 + scale(idx) | playerID),
  data = d.revisions,
  control = lmerControl(optimizer = "bobyqa")
)

summary(mod_adaptive)

# Categorize for visualization
d.plot <- d.revisions %>%
  mutate(
    # Categorize sample size (tertiles)
    sample_size_cat = case_when(
      sample_size <= 5 ~ "Small sample\n(≤5 critters)",
      sample_size <= 8 ~ "Medium sample\n(6-8 critters)",
      TRUE ~ "Large sample\n(≥9 critters)"
    ),
    sample_size_cat = factor(sample_size_cat,
                             levels = c("Small sample\n(≤5 critters)",
                                       "Medium sample\n(6-8 critters)",
                                       "Large sample\n(≥9 critters)")),
    # Categorize social discrepancy (median split)
    discrepancy_cat = case_when(
      prior_social_discrepancy <= median(prior_social_discrepancy, na.rm = TRUE) ~ "Low discrepancy",
      TRUE ~ "High discrepancy"
    ),
    discrepancy_cat = factor(discrepancy_cat, levels = c("Low discrepancy", "High discrepancy"))
  )

# Calculate overall means and 95% CIs using tidyboot (aggregated across rounds)
d.summary <- d.plot %>%
  group_by(sample_size_cat, discrepancy_cat) %>%
  tidyboot_mean(belief_revision, na.rm = TRUE)

# Bar chart
p_adaptive <- ggplot(d.summary,
                     aes(x = sample_size_cat, y = empirical_stat,
                         fill = discrepancy_cat)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75, alpha = 0.9) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                position = position_dodge(width = 0.8),
                width = 0.25, linewidth = 0.6) +
  scale_fill_manual(
    name = "Social Discrepancy",
    values = c("Low discrepancy" = "#0072B2", "High discrepancy" = "#D55E00"),
    labels = c("Low (similar to others)", "High (divergent from others)")
  ) +
  scale_y_continuous(limits = c(0, 14), breaks = seq(0, 14, 2), expand = c(0, 0)) +
  labs(
    x = "Sample Size",
    y = "Mean Belief Revision\n(absolute change in estimate)"
  ) +
  theme_few(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    panel.grid.major.y = element_line(color = "gray90"),
    legend.box = "vertical",
    legend.margin = margin(t = 10)
  )

# Save figure
ggsave(here("figures/fig3.pdf"), p_adaptive,
       width = 10, height = 6)

p_adaptive

# Print summary statistics
cat("\nSummary statistics by group:\n")
print(d.summary)

cat("\n\nEffect sizes:\n")
small_diff <- d.summary %>% filter(sample_size_cat == "Small sample\n(≤5 critters)")
large_diff <- d.summary %>% filter(sample_size_cat == "Large sample\n(≥9 critters)")

cat("Small sample: High vs. Low discrepancy difference =",
    round(small_diff$empirical_stat[2] - small_diff$empirical_stat[1], 2), "points\n")
cat("Large sample: High vs. Low discrepancy difference =",
    round(large_diff$empirical_stat[2] - large_diff$empirical_stat[1], 2), "points\n")
```

## Analysis 5: Guess stabilization over time

```{r}
# Calculate change in guess from round to round (using shifted data)
# Exclude slider's baseline round (idx_shifted=0) so all conditions are aligned:
# first change = transition from first info round to second info round
d.stabilization <- d.shifted %>%
  filter(idx_shifted >= 1) %>%
  arrange(playerID, idx_shifted) %>%
  group_by(playerID) %>%
  mutate(
    abs_change = abs(guess - lag(guess))
  ) %>%
  ungroup() %>%
  filter(!is.na(abs_change))

# Panel B: Guess stabilization with polynomial smooth
p_stability <- d.stabilization %>%
  group_by(condition, idx_shifted) %>%
  tidyboot_mean(abs_change, na.rm = TRUE) %>%
  ggplot(aes(x = idx_shifted, y = empirical_stat, color = condition, fill = condition)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.15, color = NA) +
    geom_point(size = 2) +
    geom_smooth(method = 'lm', formula = y ~ poly(x, 2), se = FALSE, linewidth = 1) +
    scale_color_manual(
      name = NULL,
      values = c("slider" = "#E69F00", "unidirectional" = "#56B4E9", "interactive" = "#009E73"),
      labels = c("Slider", "Unidirectional", "Interactive"),
      breaks = c("slider", "unidirectional", "interactive"),
      drop = FALSE
    ) +
    scale_fill_manual(
      name = NULL,
      values = c("slider" = "#E69F00", "unidirectional" = "#56B4E9", "interactive" = "#009E73"),
      labels = c("Slider", "Unidirectional", "Interactive"),
      breaks = c("slider", "unidirectional", "interactive"),
      drop = FALSE
    ) +
    scale_x_continuous(breaks = c(2, 5, 8, 11)) +
    labs(
      x = "Round",
      y = "Change in Guess from Previous Round",
      tag = "B"
    ) +
    theme_few(base_size = 12) +
    theme(
      legend.position = "none",
      plot.tag = element_text(face = "bold", size = 18),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14)
    )

# Combine panels (legend only in panel A, upper-right)
combined_fig <- p_error + p_stability

# Save combined figure
ggsave(here("figures/fig2.pdf"), combined_fig, width = 12, height = 5)

combined_fig

# Model: Do guesses stabilize over time?
# Linear model with slider baseline (for overall effect and condition differences)
mod_stab <- lmer(
  abs_change ~ idx_shifted * condition + (1 + idx_shifted | playerID),
  data = d.stabilization,
  control = lmerControl(optimizer = "bobyqa")
)

cat("Linear model (slider baseline):\n")
summary(mod_stab)
```

## Appendix: Exclusion criteria robustness

Re-run the key analysis (condition differences at final round) under alternative exclusion thresholds to verify results do not depend on the preregistered 80% criterion.

```{r}
# Start from the pre-exclusion data: re-load and clean
d.robust <- d.raw %>%
  filter(!is.na(treatmentName))

# Handle duplicate rows (same as above)
unique_rows_r <- d.robust %>%
  filter(treatmentName %in% c("rerun-unidirectional- 0.7", "rerun-unidirectional- 0.3")) %>%
  distinct(playerID, gameID, guess, roundID, .keep_all = TRUE)
d.robust <- d.robust %>%
  filter(!(treatmentName %in% c("rerun-unidirectional- 0.7", "rerun-unidirectional- 0.3"))) %>%
  bind_rows(unique_rows_r)

d.robust <- d.robust %>%
  mutate(
    treatmentName = case_when(
      treatmentName == "rerun-unidirectional- 0.7" ~ 'unidirectional-0.7',
      treatmentName == "rerun-unidirectional- 0.3" ~ 'unidirectional-0.3',
      treatmentName == "rerun-interactive-0.3" ~ 'interactive-0.3',
      treatmentName == "rerun-interactive- 0.7" ~ 'interactive-0.7',
      treatmentName == "rerun-slider- 0.3" ~ 'slider-0.3',
      treatmentName == "rerun-slider- 0.7" ~ 'slider-0.7',
      TRUE ~ treatmentName
    )
  ) %>%
  separate(treatmentName, into = c('condition', 'treatment'), sep = '-') %>%
  mutate(
    error = abs(guess - mleEstimateGame),
    condition = factor(condition, levels = c("slider", "unidirectional", "interactive"))
  ) %>%
  arrange(playerID, idx)

# Response rates for all participants
response_rates <- d.robust %>%
  group_by(playerID) %>%
  summarise(response_rate = n_distinct(idx) / 12, .groups = "drop")

# Test each threshold
thresholds <- c(0, 0.5, 0.8, 1.0)
threshold_labels <- c("None", "50%", "80% (preregistered)", "100%")

robustness_results <- list()

for (i in seq_along(thresholds)) {
  thresh <- thresholds[i]

  if (thresh == 0) {
    d.sub <- d.robust
  } else {
    exclude_ids <- response_rates %>%
      filter(response_rate < thresh) %>%
      pull(playerID)
    d.sub <- d.robust %>% filter(!(playerID %in% exclude_ids))
  }

  # Create shifted data with interactive as reference
  d.sub <- d.sub %>%
    mutate(
      idx_shifted = if_else(condition == "slider", idx - 1L, idx),
      idx_final_centered = if_else(condition == "slider",
                                     idx_shifted - 11, idx_shifted - 12),
      condition_interactive = relevel(condition, ref = "interactive")
    )

  mod <- lmer(
    error ~ poly(idx_final_centered, 2) * condition_interactive +
      (1 + idx_final_centered | playerID),
    data = d.sub,
    control = lmerControl(optimizer = "bobyqa")
  )

  s <- coef(summary(mod))

  robustness_results[[i]] <- tibble(
    threshold = threshold_labels[i],
    n_participants = n_distinct(d.sub$playerID),
    n_networks = n_distinct(d.sub$gameID),
    slider_b = round(s["condition_interactiveslider", "Estimate"], 2),
    slider_t = round(s["condition_interactiveslider", "t value"], 2),
    slider_p = s["condition_interactiveslider", "Pr(>|t|)"],
    unidir_b = round(s["condition_interactiveunidirectional", "Estimate"], 2),
    unidir_t = round(s["condition_interactiveunidirectional", "t value"], 2),
    unidir_p = s["condition_interactiveunidirectional", "Pr(>|t|)"]
  )
}

robustness_df <- bind_rows(robustness_results)

cat("=== EXCLUSION CRITERIA ROBUSTNESS ===\n\n")
cat("Slider vs Interactive (at final round):\n")
for (i in seq_len(nrow(robustness_df))) {
  r <- robustness_df[i, ]
  cat(sprintf("  %s (N=%d, %d networks): b=%.2f, t=%.2f, p=%s\n",
              r$threshold, r$n_participants, r$n_networks,
              r$slider_b, r$slider_t, format.pval(r$slider_p, digits = 3)))
}

cat("\nUnidirectional vs Interactive (at final round):\n")
for (i in seq_len(nrow(robustness_df))) {
  r <- robustness_df[i, ]
  cat(sprintf("  %s (N=%d): b=%.2f, t=%.2f, p=%s\n",
              r$threshold, r$n_participants,
              r$unidir_b, r$unidir_t, format.pval(r$unidir_p, digits = 3)))
}
```

## Save cleaned data for figures

```{r}
write_csv(d.guesses, here("analysis/exp1/exp1_clean_data.csv"))
cat("Saved cleaned data to analysis/exp1/exp1_clean_data.csv\n")
```
