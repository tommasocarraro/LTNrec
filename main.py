from ltnrec.evaluation import experiment
from ltnrec.data import MindReaderDataset
from ltnrec.utils import generate_report_dict, generate_report_table

if __name__ == '__main__':
    data = MindReaderDataset(data_path="./dataset/mr_ntp_all_entities")

    experiment(models=["mf", "ltn", "ltn-genres"], data=data, proportions=[1.0, 0.8, 0.6, 0.4, 0.2],
              test_metrics=["hit@10", "ndcg@10"], report_save_path="./results/results.json", starting_seed=0,
               n_runs=2, parallelize=True)

    generate_report_dict("./results/results.json", "./results/results_summary.json")

    table = generate_report_table("./results/results_summary.json")
    print(table)
