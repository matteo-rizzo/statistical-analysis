from functional.stats import statistical_adjustments


def main():
    p_values = [[0.008, 0.001, 0.002, 0.004, 0.017, 0.020, 0.000, 0.003, 0.010]]
    experiments = ["learned_vs_uniform"]
    data = {k: v for k, v in zip(experiments, p_values)}
    statistical_adjustments(data)


if __name__ == '__main__':
    main()
