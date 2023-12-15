library(tidyverse)
library(ggpubr)

data <- read_csv("SemEval2024-task8/system/info_red_a_train_mono.csv") %>%
  pivot_longer(cols = c(info_red_loss_norm, info_red_trunc_max, info_red_trunc_min,
                        info_red_trunc_mean, info_red_trunc_median, info_red_sum_diff),
               names_to = "stat",
               values_to = "value") %>%
  mutate(label = factor(label)) %>%
  mutate(lower = quantile(value, probs=0.025),
         upper = quantile(value, probs=0.975)) %>%
  filter(value >= lower, value <= upper) %>%
  mutate(label = case_match(label, "0" ~ "human", "1" ~ "machine"),
         stat = case_match(stat,
                           "info_red_loss_norm" ~ "Information loss (norm of difference)",
                           "info_red_trunc_max" ~ "Max value of truncated form",
                           "info_red_trunc_min" ~ "Min value of truncated form",
                           "info_red_trunc_mean" ~ "Mean value of truncated form",
                           "info_red_trunc_median" ~ "Median value of truncated form",
                           "info_red_sum_diff" ~ "Information loss (sum of difference)"))
p <- data %>% gghistogram(x = "value", fill = "label", facet.by = "stat", scales = "free", add = "mean")
p1 <- data %>% ggviolin(x = "label", y = "value", facet.by = "stat", scales = "free")