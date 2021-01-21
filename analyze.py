from math import sqrt

from numpy import mean
from numpy import var
from statsmodels.stats.multitest import multipletests


def statistical_adjustments():
    """
    Benjamini–Hochberg, 1995 <https://www.jstor.org/stable/2346101?seq=1>

    We account for multiple comparisons using the Benjamini–Hochberg procedure to control for the false discovery rate:
        1. The obtained p-values from our models are ordered from smallest to largest, such that the smallest p value has a
           rank of i=1, the next smallest has i=2, etc.
        2. We compare each individual p-value to its Benjamini–Hochberg critical threshold of q = (i / m) * α, where:
            - i is the rank
            - m is the total number of models
            - α is set to 0.05
        3. We find the largest p-value that has p<q given its rank r.
        4. All p-values at rank i≤r are considered significant.
    """

    # experiments = {
    #     "3-folds": [0.2007, 0.0250, 0.0565, 0.3892, 0.0413, 0.1131],
    #     "4-folds": [0.0923, 0.0045, 0.0198, 0.1870, 0.0339, 0.0430],
    # }

    experiments = {
        "4-folds": [0.2767, 0.1219, 0.1437, 0.4227, 0.0690, 0.2350],
    }

    experiments = {
        "4-folds": [0.0923,	0.0045,	0.0198],
    }

    print(" *** Benjamini–Hochberg Analysis *** ")

    print("\n---------------------------------------")

    for exp_id, pvals in experiments.items():
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

        print("\n Results for {}:".format(exp_id))
        print("\t * p-values (input): ....... {}".format(pvals))
        print("\t * p-values (corrected): ... {}".format(pvals_corrected))
        print("\t * Reject: ................. {}".format(reject))

    print("\n---------------------------------------\n")


def cohend(d1: list, d2: list) -> float:
    """ Calculates Cohen's d for two sets of independent samples """

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


def effect_size():
    """
    Cohen’s d measures the difference between the mean from two Gaussian-distributed variables.
    It is a standard score that summarizes the difference in terms of the number of standard deviations.
    Because the score is standardized, there is a table for the interpretation of the result, summarized as:
        * Small: d=0.20
        * Medium: d=0.50
        * Large: d=0.80
    """

    # experiments = {
    #     "3-folds": {
    #         "tccnet": [1.88, 1.82, 2.22],
    #         "tccnetc4": [1.70, 1.85, 2.17],
    #         "ctccnet": [1.85, 1.76, 2.18],
    #         "ctccnetc4": [1.62, 1.72, 2.13]
    #     },
    #     "4-folds": {
    #         "tccnet": [1.99, 1.88, 1.82, 2.22],
    #         "tccnetc4": [1.72, 1.70, 1.85, 2.17],
    #         "ctccnet": [1.95, 1.85, 1.76, 2.18],
    #         "ctccnetc4": [1.70, 1.62, 1.72, 2.13]
    #     }
    # }

    experiments = {
        "4-folds": {
            "tccnet": [1.46, 1.25, 1.27, 1.53],
            "tccnetc4": [1.20, 1.16, 1.29, 1.64],
            "ctccnet": [1.42, 1.27, 1.24, 1.44],
            "ctccnetc4": [1.13, 0.98, 1.28, 1.60]
        }
    }

    print(" *** Effect Size Analysis (Cohen's d) *** ")

    print("\n---------------------------------------")

    for exp_id, exp in experiments.items():
        print("\nResults for {}:".format(exp_id))
        print("\t [ TCCNet    vs TCCNetC4  ] : {:.4f}".format(cohend(exp["tccnet"], exp["tccnetc4"])))
        print("\t [ TCCNet    vs CTCCNet   ] : {:.4f}".format(cohend(exp["tccnet"], exp["ctccnet"])))
        print("\t [ TCCNet    vs CTCCNetC4 ] : {:.4f}".format(cohend(exp["tccnet"], exp["ctccnetc4"])))
        print("\t [ TCCNetC4  vs CTCCNetC4 ] : {:.4f}".format(cohend(exp["tccnetc4"], exp["ctccnetc4"])))
        print("\t [ CTCCNet   vs TCCNetC4  ] : {:.4f}".format(cohend(exp["ctccnet"], exp["tccnetc4"])))
        print("\t [ CTCCNet   vs CTCCNetC4 ] : {:.4f}".format(cohend(exp["ctccnet"], exp["ctccnetc4"])))

    print("\n---------------------------------------\n")


def main():
    effect_size()
    statistical_adjustments()


if __name__ == '__main__':
    main()
