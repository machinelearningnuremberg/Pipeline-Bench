from __future__ import annotations

import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn import preprocessing
from pathlib import Path

import openml


class Dataset:
    def __init__(
        self,
        task_id: int,
        worker_dir: str | Path,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.task_id = task_id
        self.random_state = random_state

        self.worker_dir = Path(worker_dir).parent
        self._fetch_and_save_datasets_names_and_ids(worker_dir=self.worker_dir)

        sklearn_datasets = pd.read_csv(self.worker_dir / "sklearn_datasets.csv")
        openml_datasets = pd.read_csv(self.worker_dir / "openml_cc18_datasets.csv")

        self.benchmark_suite = None
        if task_id in sklearn_datasets["task_id"].values:  # sklearn dataset
            dataset_name = sklearn_datasets[sklearn_datasets["task_id"] == task_id][
                "name"
            ].values[0]
            self.data: tuple = getattr(sklearn.datasets, f"load_{dataset_name}")(
                return_X_y=True
            )
            self.benchmark_suite = "sklearn"
        elif task_id in openml_datasets["task_id"].values:  # OpenML-CC18 dataset
            dataset_name = openml_datasets[openml_datasets["task_id"] == task_id][
                "name"
            ].values[0]
            self.data = sklearn.datasets.fetch_openml(
                name=dataset_name, return_X_y=True, as_frame=False
            )
            self.benchmark_suite = "OpenML-CC18"
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.name = dataset_name

        self._splits = None

    @staticmethod
    def _fetch_and_save_datasets_names_and_ids(worker_dir):
        sklearn_datasets_path = worker_dir / "sklearn_datasets.csv"
        openml_datasets_path = worker_dir / "openml_cc18_datasets.csv"

        # Check if sklearn_datasets.csv exists, if not fetch and save sklearn dataset names and ids
        if not sklearn_datasets_path.is_file():
            # Save sklearn datasets names and ids
            sklearn_datasets = pd.DataFrame(
                {
                    "task_id": [0, -1, -2, -3],
                    "name": ["breast_cancer", "iris", "wine", "digits"],
                }
            )
            sklearn_datasets.to_csv(sklearn_datasets_path, index=False)

        # Check if openml_cc18_datasets.csv exists, if not fetch and save OpenML-CC18 dataset names and ids
        if not openml_datasets_path.is_file():
            # Save OpenML-CC18 datasets names and ids
            # get tasks in the form of a dataframe
            tasks = openml.tasks.list_tasks(output_format="dataframe")
            # get the suite's tasks
            suite = openml.study.get_suite("OpenML-CC18")
            # filter tasks that are part of the suite
            openml_cc18_tasks = tasks[tasks.tid.isin(suite.tasks)]
            # extract task ids and names
            openml_datasets = pd.DataFrame(
                {"task_id": openml_cc18_tasks.tid, "name": openml_cc18_tasks.name}
            )
            # Save OpenML-CC18 datasets names and ids
            openml_datasets.to_csv(openml_datasets_path, index=False)

    @property
    def splits(self):
        if self._splits is None:
            self._splits = self._get_split()
        return self._splits

    def get_data(self, split_name):
        if split_name not in {"train", "valid", "test"}:
            raise ValueError(
                "Invalid split name. Expected one of: {'train', 'valid', 'test'}"
            )

        return self.splits[f"X_{split_name}"], self.splits[f"y_{split_name}"]

    # ...
    def _get_split(
        self,
        train_size: float = 0.6,
        valid_size: float = 0.2,
    ) -> dict:
        """Split the dataset into train, validation and test sets."""
        X, y = self.data

        y = preprocessing.LabelEncoder().fit_transform(y)

        X_train, X_other, y_train, y_other = sklearn.model_selection.train_test_split(
            X, y, train_size=train_size, random_state=self.random_state
        )
        valid_size_adjusted = valid_size / (1 - train_size)
        X_valid, X_test, y_valid, y_test = sklearn.model_selection.train_test_split(
            X_other,
            y_other,
            train_size=valid_size_adjusted,
            random_state=self.random_state,
        )

        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
        }
