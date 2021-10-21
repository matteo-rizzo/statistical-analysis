from functional.stats import effect_size


def main():
    # --- Full sequences ---

    # Effect size for trimean of errors
    experiments = {
        "3-folds": {
            "base": [1.73, 2.11, 1.92],
            "random": [2.69, 2.47, 3.12]
        }
    }
    effect_size(experiments)


if __name__ == '__main__':
    main()
