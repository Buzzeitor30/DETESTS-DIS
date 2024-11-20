from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
from pyevall.reports.reports import (
    PyEvALLReport,
    PyEvALLMetaReport,
)
import pandas as pd


#############################################
# Setting parameters and evaluator
#############################################
def get_metrics_from_each_task_and_approach(task_id, approach):
    return (
        {
            "stereotype": {
                "hard": ["ICM", "ICMNorm", "FMeasure", "Precision", "Recall"],
                "soft": ["ICMSoft", "ICMSoftNorm", "CrossEntropy"],
            },
            "implicit": {
                "hard": ["ICM", "ICMNorm", "FMeasure"],
                "soft": ["ICMSoft", "ICMSoftNorm", "CrossEntropy"],
            },
        }
        .get(task_id)
        .get(approach)
    )


def set_parameters_for_pyevall(task_id):
    params = {
        "stereotype": {},
        #"implicit": {"YES": ["DIRECT", "REPORTED", "JUDGEMENTAL"], "NO": []},
        "implicit": {}
    }.get(task_id)
    params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
    return params


def create_evaluator():
    evaluator = PyEvALLEvaluation()
    return evaluator


def read_json_file(file_path):
    return pd.read_json(file_path, orient="records")


#############################################
# Evaluate files
#############################################
def evaluate_task_for_a_given_list_of_files_and_approach(
    preds_files, gold_file, task_id, approach, current_gold_file_path
):
    report = PyEvALLReport()
    meta_report = PyEvALLMetaReport()
    evaluator = create_evaluator()
    metrics = get_metrics_from_each_task_and_approach(task_id, approach)
    params = set_parameters_for_pyevall(task_id)

    evaluator.load_evaluation_conf(**params)

    for num, pred in enumerate(preds_files, start=1):
        gold_df = read_json_file(gold_file)
        pred_df = read_json_file(pred)
        gold_df = create_new_gold_df_from_pred_df(pred_df, gold_df)

        gold_df.to_json(current_gold_file_path, orient="records")
        report = evaluator.evaluate(pred, current_gold_file_path, metrics, **params)

        meta_report.add_pyevall_report(report, num)

    evaluator.remove_active_evaluation()
    meta_report.print_report()
    df = from_report_to_dataframe(meta_report.report)
    df = add_average_metrics_to_results(df, task_id, approach)
    return df


def evaluate_task_hard_metrics(preds_files, gold_file, task_id, current_gold_file_path):
    preds_files = filter(lambda x: "_hard" in x, preds_files)
    return evaluate_task_for_a_given_list_of_files_and_approach(
        preds_files, gold_file, task_id, "hard", current_gold_file_path
    )


def evaluate_task_soft_metrics(preds_files, gold_file, task_id, current_gold_file_path):
    preds_files = filter(lambda x: "_soft" in x, preds_files)
    return evaluate_task_for_a_given_list_of_files_and_approach(
        preds_files, gold_file, task_id, "soft", current_gold_file_path
    )


def evaluate_task(preds_files, gold_file, task_id, current_gold_file_path):
    gold_file_hard = gold_file + "_hard.json"
    gold_file_soft = gold_file + "_soft.json"
    hard_df = evaluate_task_hard_metrics(
        preds_files, gold_file_hard, task_id, current_gold_file_path
    )
    soft_df = evaluate_task_soft_metrics(
        preds_files, gold_file_soft, task_id, current_gold_file_path
    )
    return hard_df, soft_df


#############################################
# Utils for handling dataframes
#############################################
def create_new_gold_df_from_pred_df(pred_df, gold_df):
    gold_df = gold_df[gold_df["id"].isin(pred_df["id"])]
    gold_df["id"] = gold_df["id"].astype(str, copy=True).values
    return gold_df


def from_report_to_dataframe(
    report,
):
    metrics_data = []
    for eval_key, eval_value in report.items():
        for metric_key, metric_value in eval_value["metrics"].items():
            acronym = metric_value["acronym"]
            description = metric_value["description"]
            status = metric_value["status"]
            for test_case in metric_value["results"]["test_cases"]:
                test_case_name = test_case["name"]
                average = test_case["average"]
                classes = test_case.get("classes", {})
                metrics_data.append(
                    {
                        "evaluation": eval_key,
                        # "metric": metric_key,
                        "metric": acronym,
                        # "description": description,
                        # "status": status,
                        # "test_case_name": test_case_name,
                        "average": average,
                        **classes,
                    }
                )
    return pd.DataFrame(metrics_data)


def add_average_metrics_to_results(results, task_id, approach):
    classes = ["average"]
    # todo for rest of tasks
    if approach == "hard":
        if task_id == "stereotype":
            classes = ["average", "Stereotype", "NoStereotype"]

    metric_averages = results.groupby("metric")[classes].mean().reset_index()
    metric_averages["evaluation"] = "average"
    # Reorder columns
    cols = ["evaluation", "metric"] + classes
    metric_averages = metric_averages[cols]
    # Add the average row to the original DataFrame
    df = pd.concat([results, metric_averages], ignore_index=True)
    return df
