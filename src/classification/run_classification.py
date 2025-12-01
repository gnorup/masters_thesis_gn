import os

from config.config import Config
from config.constants import GIT_DIRECTORY
from classification.classification import (
    classification_hyperparameter_tuning,
    run_all_models,
    plot_all_roc_curves,
    plot_all_pr_curves,
    plot_all_pr_curves_bootstrapped,
    CL_RESULTS_DIR,
)

# config
CLASSIFICATION_TASKS = ["picnicScene"]
CLASSIFICATION_MODELS = (
    "demographics",
    "acoustic",
    "linguistic",
    "linguistic+acoustic",
    "full",
)

# run full classification pipeline using settings from config.yaml
def run_full_classification(config):

    run_flags = config.run
    settings = config.settings
    cls_cfg = config.hyperparameter_tuning.classification

    targets = settings.targets

    # 1) optional hyperparameter tuning
    if run_flags.hyperparam_tuning_classification:
        print("\n[classification] Running classification hyperparameter tuning...")
        classification_hyperparameter_tuning(
            task_name=cls_cfg.task_name,
            target=cls_cfg.target,
            n_iter_random=cls_cfg.n_iter_random,
            refine_with_grid=cls_cfg.refine_with_grid,
            test_half_label=cls_cfg.test_half_label,
        )
        print("[classification] Classification hyperparameter tuning finished.\n")

    # 2) classification
    print("\n[classification] Running Random Forest classification...")
    for task in CLASSIFICATION_TASKS:
        for target in targets:
            print(f"[classification] {task} – {target}")
            _ = run_all_models(
                task=task,
                target=target,
                models=CLASSIFICATION_MODELS,
                outdir=CL_RESULTS_DIR,
            )
    print("[classification] Main classification analyses finished.\n")

    # 3) create plots (ROC & PR curves) for all models
    print("\n[classification] Creating ROC/PR plots...")
    for task in CLASSIFICATION_TASKS:
        for target in targets:
            plot_all_roc_curves(
                task=task,
                target=target,
                models=CLASSIFICATION_MODELS,
                outdir=CL_RESULTS_DIR,
                show_ci="one",
                ci_model="full",
                legend_metric="mean",
            )

            plot_all_pr_curves(
                task=task,
                target=target,
                models=CLASSIFICATION_MODELS,
                outdir=CL_RESULTS_DIR,
                legend_metric="mean",
                show_f1_for="full",
            )

            plot_all_pr_curves_bootstrapped(
                task=task,
                target=target,
                models=CLASSIFICATION_MODELS,
                outdir=CL_RESULTS_DIR,
                legend_metric="mean",
                show_ci="one",
                ci_model="full",
            )

    print("[classification] Overall ROC/PR plots finished.\n")

    print("\nall classification analyses completed")


if __name__ == "__main__":
    config_path = os.path.join(GIT_DIRECTORY, "src", "config", "config.yaml")
    config = Config.from_yaml(config_path)
    run_full_classification(config)
