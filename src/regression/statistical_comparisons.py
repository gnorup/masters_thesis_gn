import os
import pandas as pd

from regression.model_evaluation import (
    compare_scores,
    compare_tasks,
    compare_models,
)
from regression.plots import (
    plot_model_comparison_boxplots_bootstrapped,
    plot_bootstrap_models_tasks,
    plot_task_comparison_boxplots_bootstrapped,
    plot_score_comparison_boxplots_bootstrapped,
)
from data_preparation.data_handling import subject_intersection_for_score

from config.constants import SCORES, TASKS, MODELS, RANDOM_STATE, N_BOOT


# run score, task and model comparisons
def run_statistical_comparisons(
    oof_path,
    results_path,
    main_model="full",
    main_task="picnicScene",
    score_order=tuple(SCORES),
    task_order=tuple(TASKS),
    model_order=tuple(MODELS),
    n_boot=N_BOOT,
    random_state=RANDOM_STATE,
    make_plots=True,
):
    """
    Runs statistical comparisons:
    - pairwise score comparisons (within task)
    - pairwise task comparisons (within score)
    - pairwise model comparisons (within task & score)
    Saves tables and (optionally) plots.
    """

    # load regression results
    oof_all = pd.read_csv(oof_path)
    os.makedirs(results_path, exist_ok=True)

    ### score comparisons

    score_path = os.path.join(results_path, "score_comparison")
    os.makedirs(score_path, exist_ok=True)

    # pairwise score tests within picnicScene (full model) (comparison discussed in thesis)
    score_tests = compare_scores(
        oof_preds=oof_all,
        task=main_task,
        model=main_model,
        targets=tuple(score_order),
        n_boot=n_boot,
        adjust="holm",
        random_state=random_state,
    )
    score_tests.to_csv(
        os.path.join(score_path, f"score_comparison_{main_model}model_{main_task}.csv"),
        index=False,
    )

    if make_plots:
        # plot score comparisons for all tasks with significant p-values
        plot_score_comparison_boxplots_bootstrapped(
            oof_preds=oof_all,
            model=main_model,
            n_boot=n_boot,
            order_scores=list(score_order),
            order_tasks=list(task_order),
            adjust="holm",
            alpha=0.05,
            save_path=score_path,
        )

    # full score-comparison table for all tasks and score pairs
    all_score_rows = []
    for t in task_order:
        df_scores = compare_scores(
            oof_preds=oof_all,
            task=t,
            model=main_model,
            targets=tuple(score_order),
            n_boot=n_boot,
            adjust="holm",
            random_state=random_state,
        )
        df_scores["task"] = t
        all_score_rows.append(df_scores)

    if all_score_rows:
        all_scores_combined = pd.concat(all_score_rows, ignore_index=True)
        all_scores_combined.to_csv(
            os.path.join(score_path, f"score_comparison_{main_model}model_all_tasks.csv"),
            index=False,
        )

    ### task comparisons

    task_path = os.path.join(results_path, "task_comparison")
    os.makedirs(task_path, exist_ok=True)

    # full task-comparison table for all scores
    task_tests = compare_tasks(
        oof_all,
        scores=tuple(score_order),
        tasks=tuple(task_order),
        model=main_model,
        n_boot=n_boot,
        adjust="holm",
        random_state=random_state,
    )
    task_tests.to_csv(
        os.path.join(task_path, f"task_comparison_{main_model}model_all_scores.csv"),
        index=False,
    )

    if make_plots:
        # plot task comparisons with p-values for Picture Naming
        plot_task_comparison_boxplots_bootstrapped(
            oof_preds=oof_all,
            model=main_model,
            n_boot=n_boot,
            order_scores=score_order,
            order_tasks=task_order,
            adjust="holm",
            alpha=0.05,
            score_for_brackets="PictureNamingScore",
            save_path=task_path,
        )

        # plot task comparisons with p-values for Semantic Fluency
        plot_task_comparison_boxplots_bootstrapped(
            oof_preds=oof_all,
            model=main_model,
            n_boot=n_boot,
            order_scores=score_order,
            order_tasks=task_order,
            adjust="holm",
            alpha=0.05,
            score_for_brackets="SemanticFluencyScore",
            save_path=task_path,
        )

    ### model comparisons

    model_path = os.path.join(results_path, "model_comparison")
    os.makedirs(model_path, exist_ok=True)

    # pairwise model tests for Semantic Fluency (picnicScene) (comparison discussed in thesis)
    model_tests_sf = compare_models(
        oof_all,
        task=main_task,
        target="SemanticFluencyScore",
        models=model_order,
        n_boot=n_boot,
        adjust="holm",
        random_state=random_state,
    )
    model_tests_sf.to_csv(
        os.path.join(model_path, f"model_comparison_{main_task}_SemanticFluency.csv"),
        index=False,
    )

    if make_plots:
        # model comparison plot with selected p-values to show
        plot_model_comparison_boxplots_bootstrapped(
            oof_all,
            target="SemanticFluencyScore",
            task=main_task,
            order_models=model_order[1:],  # drop baseline
            pairs_to_show=None, # e.g., [("demographics", "acoustic"), ("demographics", "linguistic")]
            tests_df=model_tests_sf,
            save_path=model_path,
        )

    # full model comparison for all tasks and scores
    all_model_rows = []
    for t in task_order:
        for target in score_order:
            df_models = compare_models(
                oof_all,
                task=t,
                target=target,
                models=model_order,
                n_boot=n_boot,
                adjust="holm",
                random_state=random_state,
            )
            all_model_rows.append(df_models)

    if all_model_rows:
        model_comparisons_all = pd.concat(all_model_rows, ignore_index=True)
        model_comparisons_all.to_csv(
            os.path.join(model_path, "model_comparison_all_tasks_all_scores.csv"),
            index=False,
        )

    if make_plots:
        # additional plots without p-values: boxplot for each score, all tasks, all models
        for target in score_order:
            oof_target = oof_all[oof_all["target"] == target].copy()
            subject_set = subject_intersection_for_score(
                oof_all=oof_all,
                target=target,
                tasks=task_order,
                models=model_order[1:],
            )

            plot_bootstrap_models_tasks(
                oof_target,
                target=target,
                order_models=model_order[1:],  # drop baseline
                order_tasks=task_order,
                subject_set=subject_set,
                save_path=model_path,
            )
