import os

import pandas as pd

from stats import statistical_adjustments, effect_size, anova


def main():
    # --- Full sequences ---

    # Statistical adjustments of p-values for mean error
    full_mean_err = {
        "3-folds": [0.2007, 0.0250, 0.0565, 0.3892, 0.0413, 0.1131],
        "4-folds": [0.0923, 0.0045, 0.0198, 0.1870, 0.0339, 0.0430],
    }
    statistical_adjustments(experiments=full_mean_err)

    # Effect size for trimean of errors
    full_trimean_err = {
        "4-folds": {
            "tccnet": [2.41, 2.34, 2.58, 3.18],
            "tccnetc4": [2.41, 2.54, 2.70, 2.38],
            "ctccnet": [2.41, 2.19, 2.57, 3.03],
            "ctccnetc4": [2.61, 2.63, 2.63, 2.33]
        }
    }
    effect_size(experiments=full_trimean_err)

    # --- Impact of sequence length --

    # Mean error
    data = pd.read_csv(os.path.join("assets", "seq_len_impact.csv"))
    target_vars = ("Mean", "Length")
    print("\n Impact of sequence length on {} ~ {} \n".format(target_vars[0], target_vars[1]))
    for model_id in ["tccnet", "tccnetc4", "ctccnet", "ctccnetc4"]:
        print("\n Model '{}' \n".format(model_id))
        anova(data=data[data["Model"] == model_id], vars=target_vars)


if __name__ == '__main__':
    main()
