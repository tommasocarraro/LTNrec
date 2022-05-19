import json
import numpy as np


def generate_report_dict(results_path, save_path):
    """
    Generate a report dictionary (mean and variance) given the results produced by the experiments.

    :param results_path: path where to take the results of the experiments
    :param save_path: path where to save the report dictionary as json file
    :return:
    """
    with open(results_path) as fp:
        results = json.load(fp)

    results_dict = {}
    for model in results:
        results_dict[model] = {}
        for seed in results[model]:
            for p in results[model][seed]:
                if p not in results_dict[model]:
                    results_dict[model][p] = {m: [v] for m, v in results[model][seed][p].items()}
                else:
                    for m, v in results[model][seed][p].items():
                        results_dict[model][p][m].append(v)

    for model in results_dict:
        for fold in results_dict[model]:
            for m, v in results_dict[model][fold].items():
                results_dict[model][fold][m] = str(np.mean(results_dict[model][fold][m])) + " +/- " + \
                                               str(np.std(results_dict[model][fold][m]))

    with open(save_path, 'w') as fp:
        json.dump(results_dict, fp, indent=4)


def generate_report_table(report_path):
    """
    Generate a report table given a report json file.

    :param report_path: path to the report json file.
    """
    with open(report_path) as f:
        report = json.load(f)

    models = list(report.keys())
    folds = list(report[models[0]].keys())
    metrics = list(report[models[0]][folds[0]].keys())

    report_dict = {fold: {metric: [] for metric in metrics} for fold in folds}

    for model in models:
        for fold in folds:
            for metric in metrics:
                report_dict[fold][metric].append((round(float(report[model][fold][metric].split(" +/- ")[0]), 4),
                                                  round(float(report[model][fold][metric].split(" +/- ")[1]), 4)))

    max_fold_metric = {fold: {metric: max([metric_mean for metric_mean, _ in report_dict[fold][metric]])
                              for metric in metrics}
                       for fold in folds}

    table = "\\begin{table*}[ht!]\n\\centering\n\\begin{tabular}{ l | l | " + " | ".join(["c" for _ in models]) + " }\n"
    table += "Fold & Metric & " + " & ".join([model for model in models]) + "\\\\\n\\hline"
    for fold in report_dict:
        table += "\n\\multirow{%d}{*}{%d\%%}" % (len(metrics), float(fold) * 100)
        for metric in metrics:
            table += " &" + (" %s & " + " & ".join([("%.4f$_{(%.4f)}$" if mean_metric != max_fold_metric[fold][metric] else "\\textbf{%.4f}$_{(%.4f)}$") % (mean_metric, variance_metric) for mean_metric, variance_metric in report_dict[fold][metric]])) % metric + "\\\\\n"
        table += "\\hline"

    table += "\n\\end{tabular}\n\\caption{Test metrics}\n\\end{table*}"
    return table
