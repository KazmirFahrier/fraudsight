# FraudSight — statistical inference in R (Phase 7)
# Bootstrap 95% CI for PR-AUC (average precision), a paired model-comparison
# bootstrap, and a calibration / reliability check. Mirrors src/evaluate.py so
# results can be cross-checked across languages.
#
# Usage:
#   Rscript R/inference.R scores.csv   # csv with columns: y, score_a, score_b
# Requires: PRROC  (install.packages("PRROC"))

suppressWarnings(suppressMessages(library(PRROC)))

average_precision <- function(y, s) {
  pr <- pr.curve(scores.class0 = s[y == 1],
                 scores.class1 = s[y == 0],
                 curve = FALSE)
  pr$auc.integral
}

boot_pr_auc_ci <- function(y, s, R = 2000, alpha = 0.05, seed = 42) {
  set.seed(seed)
  n <- length(y)
  stats <- numeric(0)
  for (i in seq_len(R)) {
    idx <- sample.int(n, n, replace = TRUE)
    if (sum(y[idx]) == 0) next            # skip degenerate resamples
    stats <- c(stats, average_precision(y[idx], s[idx]))
  }
  c(mean = mean(stats),
    lo = quantile(stats, alpha / 2, names = FALSE),
    hi = quantile(stats, 1 - alpha / 2, names = FALSE))
}

# Paired bootstrap on the DIFFERENCE: is model A really better than B, or noise?
paired_diff_ci <- function(y, sa, sb, R = 2000, seed = 42) {
  set.seed(seed)
  n <- length(y); d <- numeric(0)
  for (i in seq_len(R)) {
    idx <- sample.int(n, n, replace = TRUE)
    if (sum(y[idx]) == 0) next
    d <- c(d, average_precision(y[idx], sa[idx]) - average_precision(y[idx], sb[idx]))
  }
  ci <- quantile(d, c(0.025, 0.975), names = FALSE)
  cat(sprintf("PR-AUC(A) - PR-AUC(B): mean=%.4f  95%% CI [%.4f, %.4f]  %s\n",
              mean(d), ci[1], ci[2],
              if (ci[1] > 0) "A significantly better" else
              if (ci[2] < 0) "B significantly better" else
              "difference not significant (CI straddles 0)"))
}

# Reliability: bin predicted probs, compare to observed fraud rate per bin.
calibration_table <- function(y, s, bins = 10) {
  br <- cut(s, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)
  agg <- aggregate(list(pred = s, obs = y), by = list(bin = br), FUN = mean)
  agg$n <- as.numeric(table(br))
  agg
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1 && file.exists(args[1])) {
    df <- read.csv(args[1])
    print(boot_pr_auc_ci(df$y, df$score_a))
    if (!is.null(df$score_b)) paired_diff_ci(df$y, df$score_a, df$score_b)
    print(calibration_table(df$y, df$score_a))
  } else {
    cat("Provide a CSV with columns y, score_a[, score_b]. See header.\n")
  }
}
