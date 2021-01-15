from statsmodels.stats.multitest import multipletests

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


def main():
    experiments = {
        "3-folds": [0.2007, 0.0250, 0.0565],
        "4-folds": [0.0923, 0.0045, 0.0198],
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


if __name__ == '__main__':
    main()
