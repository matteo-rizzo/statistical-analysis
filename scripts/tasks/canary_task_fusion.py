import os

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    # data = pd.read_csv(os.path.join("assets", "task_fusion", "processed", "anova_scores.csv"))
    data = pd.read_csv(os.path.join("assets", "new_anova_scores_lang.csv"))

    # Exclude a model
    # data = data[data["model"] != "DummyClassifier"]

    # Exclude a task
    # data = data[data["task"] != "Fusion"]

    # --- ANOVA ---

    # Comparison across models for an individual task
    formula = "roc ~ model * task"

    print("\n *** ANOVA type 2 (formula: {}) *** ".format(formula))
    print("\n----------------------------------------------------------\n")
    model = ols(formula, data=data).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    print("\n----------------------------------------------------------\n")

    # -- Tukey HSD ---

    m_comp = pairwise_tukeyhsd(endog=data['roc'], groups=data.model + " / " + data.task, alpha=0.05)
    print(m_comp)

    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])

    group1_comp = tukey_data.loc[tukey_data.reject == True].groupby('group1').reject.count()
    group2_comp = tukey_data.loc[tukey_data.reject == True].groupby('group2').reject.count()
    tukey_data = pd.concat([group1_comp, group2_comp], axis=1)

    tukey_data = tukey_data.fillna(0)
    tukey_data.columns = ['reject_group_1', 'reject_group_2']
    tukey_data['total_rejections'] = tukey_data.reject_group_1 + tukey_data.reject_group_2

    print("\n-------------------------------------------------------------------------------------------------")
    print("                             Summary results of pairwise Tukey HSD")
    print("-------------------------------------------------------------------------------------------------\n")
    print(tukey_data.sort_values('total_rejections', ascending=False))


if __name__ == '__main__':
    main()
