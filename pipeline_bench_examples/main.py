import argparse
import multiprocessing
import time
import pipeline_bench


def run_pipelines(pipeline_id, task_id, worker_dir):
    try:
        benchmark = pipeline_bench.Benchmark(
            task_id=task_id,
            worker_dir=worker_dir,
            mode="live",
        )

        _ = benchmark(
            pipeline_id=pipeline_id,
        )
    except Exception as e:
        print(f"Failed for pipeline_id: {pipeline_id}")
        print(e)
        print()


def process_pipeline(pipeline_id, task_id, worker_dir):
    run_pipelines(pipeline_id, task_id, worker_dir)
    # p = multiprocessing.Process(
    #     target=run_pipelines, args=(pipeline_id, task_id, worker_dir)
    # )
    # p.start()

    # # Wait for 1200 seconds or until process finishes
    # p.join(2400)

    # # If thread is still active
    # if p.is_alive():
    #     print("Running... let's kill it...")

    #     # Terminate - may not work if process catches the signal and doesn't exit
    #     p.terminate()

    #     # Check if process has really terminated & print message
    #     time.sleep(0.1)
    #     if p.is_alive():
    #         print("Failed to terminate the process")
    #     else:
    #         print(f"Process terminated for pipeline_id: {pipeline_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_pipeline_id", type=int, required=True)
    parser.add_argument("--end_pipeline_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--worker_dir", type=str, default="/work/dlclarge1/janowski-pipebench"
    )
    args = parser.parse_args()

    for pipeline_id in range(args.start_pipeline_id, args.end_pipeline_id + 1):
        process_pipeline(pipeline_id, args.task_id, args.worker_dir)
