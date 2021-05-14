import os
import time

import pandas as pd


def main():
    base_file = "results_task_fusion_"
    tasks = ["Fusion", "CookieTheft", "Memory", "PupilCalib", "Reading"]
    path_to_results = os.path.join("assets", "task_fusion", "raw")

    path_to_processed = os.path.join("assets", "task_fusion", "processed_{}".format(time.time()))
    os.makedirs(path_to_processed)

    paths_to_seeds = [os.path.join(path_to_results, seed) for seed in sorted(os.listdir(path_to_results))]

    results = {task: [] for task in tasks}
    tasks = tasks[1:]

    print("\n *** Aggregating CV results *** \n")

    for path_to_seed in paths_to_seeds:
        print("\n Seed: {} \n".format(path_to_seed))
        print("\t -> Processing task: Fusion")
        fusion_results = pd.read_csv(os.path.join(path_to_seed, base_file + ".csv"))
        fusion_results = fusion_results.groupby(["metric", "model"], as_index=False)["1"].mean()
        results["Fusion"].append(fusion_results)

        for task in tasks:
            print("\t -> Processing task: {}".format(task))
            task_results = pd.read_csv(os.path.join(path_to_seed, "{}{}.csv".format(base_file, task)))
            task_results = task_results.groupby(["metric", "model"], as_index=False)["1"].mean()
            results[task].append(task_results)

    print("\n\n *** Aggregating SEEDS results *** \n")

    for task, seeds_results in results.items():
        print("\n\t -> Processing task: {} \n".format(task))
        agg_results = pd.concat(seeds_results).groupby(["metric", "model"], as_index=False)["1"].mean()
        agg_results.to_csv(os.path.join(path_to_processed, "{}.csv".format(task)))
        for model in agg_results["model"].unique():
            print("\t\t * Model: {}".format(model))
            path_to_model_results = os.path.join(path_to_processed, "{}_{}.csv".format(task, model))
            agg_results[agg_results["model"] == model].to_csv(path_to_model_results)


if __name__ == '__main__':
    main()
