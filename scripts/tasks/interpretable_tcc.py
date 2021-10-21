import os

import pandas as pd

from functional.stats import anova, effect_size, statistical_adjustments


def confidence_reg():
    # --- Confidence regularization --

    data = pd.read_csv(os.path.join("assets", "conf_reg.csv"))
    data = data[data["RegType"] != "Random"]

    # ~~~ Mean error ~~~

    # ANOVA for p-values
    target_vars = ("Mean", "RegType")
    print("\n Impact of regularization type on {} ~ {} \n".format(target_vars[0], target_vars[1]))
    anova(data=data, vars=target_vars)

    # Effect size
    mean_err = {}
    for reg_type in data["RegType"].unique():
        reg_data = data[data["RegType"] == reg_type]
        mean_err[reg_type] = [reg_data[reg_data["Split"] == s]["Mean"].values for s in range(3)]
    effect_size(experiments={"3-folds": mean_err})


def mean_error():
    # --- Full sequences ---

    data = pd.read_csv(os.path.join("assets", "attention_tccnet.csv"))

    # ~~~ Mean error ~~~

    # ANOVA for p-values
    target_vars = ("Mean", "Model")
    print("\n Impact of regularization type on {} ~ {} \n".format(target_vars[0], target_vars[1]))
    anova(data=data, vars=target_vars)

    print("\n Statistical adjustments of p-values for mean error on full sequences \n")

    exp = {"4-folds": [0.0632, 0.1056, 0.0170, 0.3680, 0.3562, 0.4774]}
    statistical_adjustments(experiments=exp)

    exp = {
        "4-folds": {
            "B": [2.52, 2.02, 1.98, 2.65],
            "C": [2.40, 3.21, 2.44, 3.46],
            "A": [2.45, 3.18, 2.85, 2.65],
            "CA": [2.72, 2.39, 2.78, 3.17]
        }
    }

    # Effect size for mean errors
    effect_size(experiments=exp)


def seq_len():
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


def main():
    mean_error()


if __name__ == '__main__':
    main()
