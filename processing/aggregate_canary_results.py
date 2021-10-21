import os
import time

import numpy as np
import pandas as pd


def main():
    # base_file = "results_task_fusion_"
    base_file = "results_"
    # tasks = ["Fusion", "CookieTheft", "Memory", "PupilCalib", "Reading"]
    tasks = ["new_features_Language"]
    models = ["DummyClassifier", "GausNaiveBayes", "LogReg", "RandomForest"]
    # path_to_results = os.path.join("assets", "task_fusion", "raw")
    path_to_results = os.path.join("assets", "cookie_theft", "language", "raw")

    # path_to_processed = os.path.join("assets", "task_fusion", "processed_{}".format(time.time()))
    path_to_processed = os.path.join("assets", "new_ct_lang_{}".format(time.time()))
    os.makedirs(path_to_processed)

    paths_to_seeds = [os.path.join(path_to_results, seed) for seed in sorted(os.listdir(path_to_results))]

    results = {task: [] for task in tasks}
    # tasks = tasks[1:]

    print("\n *** Aggregating CV results *** \n")

    for path_to_seed in paths_to_seeds:
        print("\n Seed: {} \n".format(path_to_seed))
        # print("\t -> Processing task: Fusion")
        # fusion_results = pd.read_csv(os.path.join(path_to_seed, base_file + ".csv"))
        # fusion_results = fusion_results.groupby(["metric", "model"], as_index=False)["1"].mean()
        # results["Fusion"].append(fusion_results)

        for task in tasks:
            print("\t -> Processing task: {}".format(task))
            task_results = pd.read_csv(os.path.join(path_to_seed, "{}{}.csv".format(base_file, task)))
            task_results = task_results.groupby(["metric", "model"], as_index=False)["1"].mean()
            results[task].append(task_results)

    print("\n\n *** Aggregating SEEDS results *** \n")

    anova_scores, agg_auc_scores = [], []
    for task, seeds_results in results.items():
        print("\n\t -> Processing task: {} \n".format(task))

        agg_results = pd.concat(seeds_results)
        auc_scores = agg_results[agg_results["metric"] == "roc"].assign(task=[task] * 10 * len(models))
        auc_scores = auc_scores.rename(columns={"1": "roc"}, inplace=False).drop("metric", axis='columns')
        auc_scores.sort_values(["model"])
        anova_scores.append(auc_scores)

        agg_results = agg_results.groupby(["metric", "model"])["1"].agg(['mean', np.std]).reset_index()
        agg_results.to_csv(os.path.join(path_to_processed, "{}.csv".format(task)), index=False)
        agg_auc_scores.append(agg_results[agg_results["metric"] == "roc"].assign(task=[task] * len(models)))

        for model in agg_results["model"].unique():
            print("\t\t * Model: {}".format(model))
            path_to_model_results = os.path.join(path_to_processed, "{}_{}.csv".format(task, model))
            agg_results[agg_results["model"] == model].to_csv(path_to_model_results)

    anova_scores = pd.concat(anova_scores).sort_values(["model", "task"])
    anova_scores.to_csv(os.path.join(path_to_processed, "anova_scores.csv"), index=False)

    pd.concat(agg_auc_scores).to_csv(os.path.join(path_to_processed, "auc_scores_agg_seeds.csv"), index=False)


if __name__ == '__main__':
    main()
