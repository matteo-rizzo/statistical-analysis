import os

import pandas as pd

from functional.stats import statistical_adjustments, effect_size, anova


def main():
    # --- Full sequences ---

    print("\n Statistical adjustments of p-values for mean error on full sequences \n")

    full_mean_err = {
        "3-folds": [0.2007, 0.0250, 0.0565, 0.3892, 0.0413, 0.1131],
        "4-folds": [0.0923, 0.0045, 0.0198, 0.1870, 0.0339, 0.0430],
    }
    statistical_adjustments(experiments=full_mean_err)

    # Effect size for trimean of errors
    full_trimean_err = {
        "4-folds": {
            "C": [2.40, 3.21, 2.44, 3.46],
            "A": [2.45, 3.18, 2.85, 2.65],
            "CA": [2.72, 2.39, 2.78, 3.17]
        }
    }
    effect_size(experiments=full_trimean_err)

    # --- Impact of sequence length --

    data = pd.read_csv(os.path.join("assets", "seq_len_impact.csv"))
    models_id = data["Model"].unique()
    lengths = data["Length"].unique()

    # ~~~ Mean error ~~~

    # ANOVA for p-values
    target_vars = ("Mean", "Length")
    print("\n Impact of sequence length on {} ~ {} \n".format(target_vars[0], target_vars[1]))
    for model_id in models_id:
        print("\n Model '{}' \n".format(model_id))
        anova(data=data[data["Model"] == model_id], vars=target_vars)

    # Effect size
    mean_err = {}
    for mid in models_id:
        mean_err[mid] = {l: data[(data["Model"] == mid) & (data["Length"] == l)]["Worst 5%"] for l in lengths}
    effect_size(experiments=mean_err)


if __name__ == '__main__':
    main()
