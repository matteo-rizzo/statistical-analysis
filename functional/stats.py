from math import sqrt
from typing import Dict, List, Tuple

import pandas as pd
from numpy import mean, var
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from utils import SEPARATOR


def statistical_adjustments(experiments: Dict):
    """
    Benjamini–Hochberg (1995)
    Reference -> <https://www.jstor.org/stable/2346101?seq=1>

    We account for multiple comparisons using the Benjamini–Hochberg procedure to control for the false discovery rate:
        1. The obtained p-values from our models are ordered from smallest to largest, such that the smallest p value has a
           rank of i=1, the next smallest has i=2, etc.
        2. We compare each individual p-value to its Benjamini–Hochberg critical threshold of q = (i / m) * α, where:
            - i is the rank
            - m is the total number of models
            - α is set to 0.05
        3. We find the largest p-value that has p<q given its rank r.
        4. All p-values at rank i≤r are considered significant.

    @param experiments: a dict with ids of the experiments as the keys and lists of p-values as the values. Example:
    experiments = {
        "3-folds": [0.2007, 0.0250, 0.0565, 0.3892, 0.0413, 0.1131],
        "4-folds": [0.0923, 0.0045, 0.0198, 0.1870, 0.0339, 0.0430],
    }
    """
    print("\n *** Benjamini–Hochberg Analysis *** ")
    print("\n{}\n".format(SEPARATOR["dashes"]))
    for exp_id, pvals in experiments.items():
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        print("\n Results for {}:".format(exp_id))
        print("\t * p-values (input): ....... {}".format(pvals))
        print("\t * p-values (corrected): ... {}".format(pvals_corrected))
        print("\t * Reject: ................. {}".format(reject))
    print("\n{}\n".format(SEPARATOR["dashes"]))


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


def effect_size(experiments: Dict):
    """
    Computes the effect sizes for the given experiments
    -> Reference: https://machinelearningmastery.com/effect-size-measures-in-python/
    @param experiments: a dict with ids of experiments as keys and "models id" - "list of metrics values for each split
    in the CV" dicts  as values. Example:
    experiments = {
        "4-folds": {
            "tccnet": [2.41, 2.34, 2.58, 3.18],
            "tccnetc4": [2.41, 2.54, 2.70, 2.38],
            "ctccnet": [2.41, 2.19, 2.57, 3.03],
            "ctccnetc4": [2.61, 2.63, 2.63, 2.33]
        }
    }
    """
    print("\n *** Effect Size Analysis (Cohen's d) *** ")
    print("\n---------------------------------------")
    for exp_id, exp in experiments.items():
        print("\n Results for {}: \n".format(exp_id))
        models_ids = exp.keys()
        for target_id in models_ids:
            for model_id, metrics in exp.items():
                if target_id == model_id:
                    continue
                print("\t [ {} vs {} ] : {:.4f}".format(target_id, model_id, cohen_d(exp[target_id], exp[model_id])))
            print()
    print("\n---------------------------------------\n")


def anova(data: pd.DataFrame, vars: Tuple):
    """
    Prints a summary table of the ANOVA including the following info:
    - sum_sq: sum of squares for model terms
    - df: degrees of freedom for model terms
    - F: F statistic value for significance of adding model terms
    - PR(>F): p-value for significance of adding model terms

    The F value in one way ANOVA is a tool to help you answer the question “Is the variance between the means of two
    populations significantly different?”. It is calculated as:
    -> F = variance of the group means (Mean Square Between) / mean of the within group variances (Mean Squared Error)

    The F value in the ANOVA test also determines the P value; The P value is the probability of getting a result at
    least as extreme as the one that was actually observed, given that the null hypothesis is true.

    A residual is computed for each value. Each residual is the difference between a entered value and the mean of all
    values for that group. A residual is positive when the corresponding value is greater than the sample mean, and is
    negative when the value is less than the sample mean. Residuals represent the portion of the variability unexplained
    by the model.

    Types of ANOVA: https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/

    Two-way ANOVA: https://www.statology.org/two-way-anova-python/

    @param data: a dataframe including the data for the analysis
    @param vars: target variables of the dataframe for an R-like formula defining the terms of the analysis
    """
    formula = "{} ~ {}".format(vars[0], vars[1])
    print("\n *** ANOVA type 2 (formula: {}) *** ".format(formula))
    print("\n---------------------------------------\n")
    model = ols(formula, data=data).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    print("\n---------------------------------------\n")
    print(MultiComparison(data[vars[0]], data[vars[1]]).tukeyhsd().summary())
