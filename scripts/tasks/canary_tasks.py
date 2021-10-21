import os
from math import sqrt
from typing import List

import pandas as pd
from numpy import mean, var
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm


def cohen_d(d1: List, d2: List) -> float:
    """
    Calculates Cohen's d for two sets of independent samples.
    -> Reference: https://machinelearningmastery.com/effect-size-measures-in-python/

    Cohen’s d measures the difference between the mean from two Gaussian-distributed variables.
    It is a standard score that summarizes the difference in terms of the number of standard deviations.
    Because the score is standardized, there is a table for the interpretation of the result, summarized as:
        * Small: d=0.20
        * Medium: d=0.50
        * Large: d=0.80
    @type d1: the first list of independent samples
    @type d2: the second list of independent samples
    @return the Cohen's d for d1 and d2
    """

    # Calculate the size of samples
    n1, n2 = len(d1), len(d2)

    # Calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)

    # Calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # Calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)

    # Calculate the effect size
    return (u1 - u2) / s


def main():
    # all_data = pd.read_csv(os.path.join("assets", "task_fusion", "processed", "anova_scores.csv"))
    all_data = pd.read_csv(os.path.join("assets", "new_anova_scores_lang.csv"))

    # Exclude a model
    # all_data = all_data[all_data["model"] != "DummyClassifier"]

    for task in all_data["task"].unique():
        print("\n*************************************************************************************************")
        print("                             PROCESSING TASK: {}".format(task))
        print("*************************************************************************************************\n")

        data = all_data[all_data["task"] == task]

        # --- ANOVA ---

        # Comparison across models for an individual task
        formula = "roc ~ model"

        print("\n *** ANOVA type 2 (formula: {}) *** ".format(formula))
        print("\n----------------------------------------------------------\n")
        model = ols(formula, data=data).fit()
        aov_table = anova_lm(model, typ=2)
        print(aov_table)
        print("\n----------------------------------------------------------\n")

        # -- Tukey HSD ---

        print(MultiComparison(data["roc"], data["model"]).tukeyhsd().summary())

        # -- Cohen's D (effect size) ---

        print("\n *** Effect size (Cohen's D) *** \n")

        checked = []
        for model1 in all_data["model"].unique():
            checked.append(model1)
            data1 = data[data["model"] == model1]["roc"].values.tolist()
            for model2 in all_data["model"].unique():
                if model2 not in checked:
                    data2 = data[data["model"] == model2]["roc"].values.tolist()
                    print("[ {} vs {} ] : {:.4f}".format(model1, model2, cohen_d(data1, data2)))


if __name__ == '__main__':
    main()
