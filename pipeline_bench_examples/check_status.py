# type: ignore
# pylint: skip-file

import os
import json
import re
import argparse
from collections import Counter, defaultdict
import pandas as pd
from prettytable import PrettyTable
from joblib import Parallel, delayed, parallel_backend, Memory

# Setup a memory location for caching purposes
location = "/tmp/joblib_cache"
memory = Memory(location, verbose=0)

# Let's make handle_exception and reading of files an atomic operation using joblib's Memory
@memory.cache
def handle_exception_and_read_file(file):
    with open(file) as f:
        error_log = json.load(f)

    try:
        exception = handle_exception(error_log["exception"])
        return exception, error_log
    except KeyError:
        print(f"Key 'exception' not found in file {file}")
        print(f"Contents of the file: {error_log}")
        return "exception_key_not_found", None


# Known exception patterns
exception_patterns = [
    (re.compile(r"has no attribute"), "Numpy_bug"),
    (re.compile(r"Unable to allocate"), "Memory_error"),
    (re.compile(r"There are significant negative eigenvalues"), "Neg_eigenvals"),
    (re.compile(r"Floating-point under-/overflow occurred at epoch"), "under-/overflow"),
    (re.compile(r"removed all features!"), "Removed_all_features"),
    (re.compile(r"Input contains NaN"), "NaN"),
    (re.compile(r"Bug in scikit-learn"), "Scikit-learn_bug"),
    (re.compile(r"invalid value encountered in matmul"), "Invalid_value_in_matmul"),
]


def handle_exception(e):
    for _, (pattern, message) in enumerate(exception_patterns):
        if pattern.search(
            e.replace("\n", " ")
        ):  # replace line breaks with spaces for pattern search
            return message
    return "other_exceptions"


def count_exceptions_and_combinations(dataset_dir):
    error_config_ids = set()

    # Initialize empty data structures to count exceptions
    classifiers_counter = defaultdict(lambda: defaultdict(int))
    preprocessors_counter = defaultdict(lambda: defaultdict(int))
    combinations_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    dataset_counter = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(dataset_dir):
        if root == os.path.join(
            dataset_dir, "errors"
        ):  # Only process files within an 'errors' subdirectory
            for file in files:
                if file.endswith(".json"):
                    exception, error_log = handle_exception_and_read_file(
                        os.path.join(root, file)
                    )
                    if error_log is None:
                        continue

                    # Identify type of exception
                    exception = handle_exception(error_log["exception"])

                    # Count exceptions for each classifier
                    classifier = error_log["config"]["classifier:__choice__"]
                    classifiers_counter[classifier][exception] += 1

                    # Count exceptions for each preprocessor
                    preprocessor = error_log["config"]["data_preprocessor:__choice__"]
                    preprocessors_counter[preprocessor][exception] += 1

                    # Count exceptions for each combination
                    combinations_counter[preprocessor][classifier][exception] += 1

                    # Count exceptions for each dataset
                    dataset_counter[os.path.basename(root)][exception] += 1

                    # Add config_id to the error set
                    error_config_ids.add(int(file.split(".")[0]))

    # Convert nested counters to pandas DataFrame
    df_classifiers = pd.DataFrame(classifiers_counter).transpose()
    df_classifiers["Total Count"] = df_classifiers.sum(axis=1)
    df_classifiers["Percentage"] = (
        df_classifiers["Total Count"] / df_classifiers["Total Count"].sum() * 100
    )

    df_preprocessors = pd.DataFrame(preprocessors_counter).transpose()
    df_preprocessors["Total Count"] = df_preprocessors.sum(axis=1)
    df_preprocessors["Percentage"] = (
        df_preprocessors["Total Count"] / df_preprocessors["Total Count"].sum() * 100
    )

    df_combinations = pd.concat(
        {
            (preprocessor, classifier): pd.Series(combinations)
            for preprocessor, classifiers in combinations_counter.items()
            for classifier, combinations in classifiers.items()
        },
        names=["Preprocessor", "Classifier"],
    )
    df_combinations = df_combinations.unstack(fill_value=0)
    df_combinations["Total Count"] = df_combinations.sum(axis=1)
    df_combinations["Percentage"] = (
        df_combinations["Total Count"] / df_combinations["Total Count"].sum() * 100
    )

    # Convert dataset_counter to DataFrame
    df_datasets = pd.DataFrame(dataset_counter).transpose()
    df_datasets["Total Count"] = df_datasets.sum(axis=1)
    for column in df_datasets.columns:
        if column != "Total Count":
            df_datasets[f"{column} Percentage"] = (
                df_datasets[column] / df_datasets["Total Count"] * 100
            )

    return (
        df_combinations,
        df_classifiers,
        df_preprocessors,
        df_datasets,
        error_config_ids,
    )


def calculate_stats_for_dataset(dataset_dir):
    config_dir = os.path.join(dataset_dir, "configurations")
    configs = len(os.listdir(config_dir)) if os.path.exists(config_dir) else 0
    configs_percentage = (configs / total_jobs) * 100

    labels_dir = os.path.join(dataset_dir, "labels")
    labels = len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0
    labels_percentage = (labels / configs) * 100 if configs else 0

    errors_dir = os.path.join(dataset_dir, "errors")
    errors = len(os.listdir(errors_dir)) if os.path.exists(errors_dir) else 0
    errors_percentage = (errors / configs) * 100 if configs else 0

    dataset_name = os.path.basename(dataset_dir)

    return [
        dataset_name,
        f"{configs} ({configs_percentage:.2f}%)",
        f"{labels} ({labels_percentage:.2f}%)",
        f"{errors} ({errors_percentage:.2f}%)",
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets.")
    parser.add_argument(
        "--worker_dir",
        type=str,
        help="The root directory to process",
        default="/work/ws/nemo/fr_mj237-pipeline_bench-0/pipeline_bench",
    )
    parser.add_argument(
        "--n_jobs", type=int, help="The number of jobs to run in parallel", default=64
    )
    args = parser.parse_args()

    # Total exceptions
    total_exceptions = Counter()
    # Total expected jobs
    total_jobs = 10000
    # Get a list of directories
    root_directory = args.worker_dir
    dirs = sorted(
        (
            d
            for d in os.listdir(root_directory)
            if os.path.isdir(os.path.join(root_directory, d))
        ),
        key=int,
    )
    dataset_dirs = [os.path.join(root_directory, dir) for dir in dirs]

    # table = PrettyTable()
    # table.field_names = ["Dataset", "Registered Jobs (Configs)", "Successful Jobs (Labels)", "Errors"]

    # with parallel_backend('loky', inner_max_num_threads=1):
    #     rows = Parallel(n_jobs=args.n_jobs)(delayed(calculate_stats_for_dataset)(dataset_dir) for dataset_dir in dataset_dirs)

    # for row in rows:
    #     table.add_row(row)

    # print(table)

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(count_exceptions_and_combinations)(dataset_dir)
            for dataset_dir in dataset_dirs[:10]
        )

    (
        dfs_combinations,
        dfs_classifiers,
        dfs_preprocessors,
        dfs_datasets,
        error_config_ids,
    ) = zip(*results)

    dfs_datasets = pd.concat(
        [
            df.assign(Dataset=dataset_dir)
            for df, dataset_dir in zip(dfs_datasets, dataset_dirs[:10])
        ],
        ignore_index=True,
    )

    print("\nDatasets:\n", dfs_datasets)
    dfs_datasets.to_csv(os.path.join(args.worker_dir, "datasets_report.csv"))
    # print("Combinations:\n", dfs_combinations)
    # print("\nClassifiers:\n", dfs_classifiers)
    # print("\nPreprocessors:\n", dfs_preprocessors)
    print("\nCommon error config_ids across all datasets:", sum(len(error_config_ids)))

    # Save the dataframes to CSV files
    # dfs_classifiers.to_csv(os.path.join(args.worker_dir, 'classifiers_report.csv'))
    # dfs_preprocessors.to_csv(os.path.join(args.worker_dir, 'preprocessors_report.csv'))
    # dfs_combinations.to_csv(os.path.join(args.worker_dir, 'combinations_report.csv'))

    # Save the pretty table to a txt file
    # with open(os.path.join(args.worker_dir, 'table_report.txt'), 'w') as f:
    #     f.write(str(table))
