import pipeline_bench
import argparse


def run_collation(task_id, worker_dir):

    benchmark = pipeline_bench.Benchmark(
        task_id=task_id,
        worker_dir=worker_dir,
        mode="table",
        lazy=True,
    )

    benchmark.collate_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--worker_dir", type=str, default="/work/ws/nemo/fr_mj237-pipeline_bench-0/"
    )
    args = parser.parse_args()

    run_collation(args.task_id, args.worker_dir)
