import os

from config.constants import GIT_DIRECTORY, TASKS
from config.config import Config

# data splitting
from data_preparation.split_data import create_splits

# feature extraction & cleaning
from feature_extraction.calculate_features import process_features
from feature_extraction.picture_description_features import main as run_picture_description_features
from data_preparation.clean_feature_sets import main as clean_all_feature_sets
from data_preparation.clean_picture_description import main as clean_all_picture_description

# regression
from regression.hyperparameter_tuning import regression_hyperparameter_tuning
from regression.random_forest_regression import run_rf_regression
from regression.multiple_linear_regression import run_multiple_linear_regression
from regression.picture_description_comparison import run_picture_description_variants_comparison
from regression.feature_importance import run_all_shap_outputs
from regression.bias_analyses import run_bias_all_scores
from regression.statistical_comparisons import run_statistical_comparisons

# correlation
from additional_analyses.correlation_analyses import run_correlation_analyses


def main():
    # load YAML config
    config_path = os.path.join(GIT_DIRECTORY, "src", "config", "config.yaml")
    config = Config.from_yaml(config_path)

    run_flags = config.run

    # paths from config.paths + GIT_DIRECTORY
    rf_predictions_path = os.path.join(
        GIT_DIRECTORY,
        config.paths.rf_predictions,
    )
    rf_results_dir = os.path.join(
        GIT_DIRECTORY,
        config.paths.rf_results_dir,
    )

    linear_predictions_path = os.path.join(
        GIT_DIRECTORY,
        config.paths.linear_predictions,
    )
    linear_results_dir = os.path.join(
        GIT_DIRECTORY,
        config.paths.linear_results_dir,
    )

    # 1) create stratified folds (5-fold + half-split)
    if run_flags.create_splits:
        print("\n[1] Creating stratified folds...")
        create_splits()
        print("[1] Stratified folds created.\n")

    # 2) feature extraction for main tasks (cookieTheft, picnicScene, journaling)
    if run_flags.calculate_features:
        print("\n[2] Running feature extraction for main tasks...")
        for task_name in TASKS:
            print(f"\n[2] Processing all subjects for task: {task_name}\n")
            process_features(task_name)
            print(f"[2] Feature extraction complete for {task_name}")
        print("[2] Main task feature extraction finished.\n")

    # 3) clean feature sets for main tasks
    if run_flags.clean_features:
        print("\n[3] Cleaning feature sets for main tasks...")
        clean_all_feature_sets()
        print("[3] Cleaning main-task feature sets finished.\n")

    # 4) picture description feature extraction (1min / 2min / ≤5min)
    if run_flags.calculate_picture_description_features:
        print("\n[4] Running picture description feature extraction (1min, 2min, ≤5min)...")
        run_picture_description_features()
        print("[4] Picture description feature extraction finished.\n")

    # 5) clean picture description feature sets
    if run_flags.clean_picture_description:
        print("\n[5] Cleaning picture description feature sets...")
        clean_all_picture_description()
        print("[5] Cleaning picture-description feature sets finished.\n")

    # 6) hyperparameter tuning for Random Forest regression
    if run_flags.hyperparam_tuning_regression:
        print("\n[6] Running Random Forest hyperparameter tuning (regression)...")
        hp = config.hyperparameter_tuning.regression
        regression_hyperparameter_tuning(
            task_name=hp.task_name,
            target=hp.target,
            model_name=hp.model_name,
            test_half_label=hp.test_half_label,
            n_iter_random=hp.n_iter_random,
            refine_with_grid=hp.refine_with_grid,
        )
        print("[6] Hyperparameter tuning (regression) finished.\n")

    # 7) Random Forest regression
    if run_flags.run_random_forest_regression:
        print("\n[7] Running Random Forest regression...")
        run_rf_regression()
        print("[7] Random Forest regression finished.\n")

    # 8) Statistical comparisons for Random Forest
    if run_flags.run_statistical_comparisons_rf:
        print("\n[8] Running statistical comparisons for Random Forest models...")
        run_statistical_comparisons(
            oof_path=rf_predictions_path,
            results_path=rf_results_dir,
        )
        print("[8] RF statistical comparisons finished.\n")

    # 9) SHAP feature-importance analyses
    if run_flags.run_shap_analyses:
        print("\n[9] Running SHAP feature-importance analyses...")
        run_all_shap_outputs()
        print("[9] SHAP analyses finished.\n")

    # 10) bias analyses for RF models
    if run_flags.run_bias_analyses:
        print("\n[10] Running bias analyses for Random Forest models...")
        bias_out_path = os.path.join(rf_results_dir, "bias")
        run_bias_all_scores(
            oof_path=rf_predictions_path,
            out_path=bias_out_path,
            task="picnicScene",
            model="full",
        )
        print("[10] Bias analyses finished.\n")

    # 11) picture description (1min / 2min / ≤5min) comparison
    if run_flags.run_picture_description_comparison:
        print("\n[11] Running picture description sample-length comparison...")
        run_picture_description_variants_comparison()
        print("[11] Picture description comparison finished.\n")

    # 12) Multiple linear regression
    if run_flags.run_linear_regression:
        print("\n[12] Running multiple linear regression...")
        run_multiple_linear_regression()
        print("[12] Multiple linear regression finished.\n")

    # 13) Statistical comparisons for linear models
    if run_flags.run_statistical_comparisons_linear:
        print("\n[13] Running statistical comparisons for linear models...")
        run_statistical_comparisons(
            oof_path=linear_predictions_path,
            results_path=linear_results_dir,
            make_plots=False,
        )
        print("[13] Linear statistical comparisons finished.\n")

    # 14) classification analyses (RF)
    if run_flags.run_classification:
        from classification.run_classification import run_full_classification
        print("\n[14] Running classification analyses (RF classifier)...")
        run_full_classification(config)
        print("[14] Classification analyses finished.\n")

    # 15) correlation analyses (descriptives)
    if run_flags.run_correlation_analyses:
        print("\n[15] Running correlation analyses (features, scores, demographics)...")
        run_correlation_analyses(
            tasks=config.settings.tasks,
            scores=config.settings.targets,
        )
        print("[15] Correlation analyses finished.\n")

    print("\nAll selected pipeline steps completed.")


if __name__ == "__main__":
    main()
