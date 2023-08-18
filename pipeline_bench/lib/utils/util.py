from __future__ import annotations

import random
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import json
import traceback
import logging
from ConfigSpace import Configuration

import os
import dask.dataframe as dd
from functools import partial
from glob import glob

# from concurrent.futures import ThreadPoolExecutor, as_completed
import regex as re
import shutil
from tqdm import tqdm

from scipy.stats import skew, kurtosis
from dataclasses import dataclass, field


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError from e


@dataclass
class MetaFeatures:
    num_datapoints: int = field(default_factory=int)
    num_classes: int = field(default_factory=int)
    num_categorical: int = field(default_factory=int)
    num_numerical: int = field(default_factory=int)
    num_missing_values: int = field(default_factory=int)
    class_balance: dict = field(default_factory=dict)
    num_zero_variance_features: int = field(default_factory=int)
    skewness: dict = field(default_factory=dict)
    kurtosis: dict = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def update(self, df: pd.DataFrame):
        self.num_datapoints = int(df.shape[0])
        self.num_classes = int(df["y"].nunique())
        self.num_categorical = int(
            df.select_dtypes(include=["object", "category"]).shape[1]
        )
        self.num_numerical = int(df.select_dtypes(include=["int64", "float64"]).shape[1])
        self.num_missing_values = int(df.isnull().sum().sum())
        self.class_balance = {
            k: int(v) for k, v in df["y"].value_counts(normalize=True).to_dict().items()
        }
        self.num_zero_variance_features = int((df.nunique() == 1).sum())

        # Skewness and kurtosis of numerical columns
        self.skewness = {
            k: float(v)
            for k, v in df.select_dtypes(include=["int64", "float64"])
            .apply(skew)
            .to_dict()
            .items()
        }
        self.kurtosis = {
            k: float(v)
            for k, v in df.select_dtypes(include=["int64", "float64"])
            .apply(kurtosis)
            .to_dict()
            .items()
        }


def default_global_seed_gen(
    rng: np.random.RandomState | None = None, global_seed: int | None = None
) -> Iterable[int]:
    if global_seed is not None:
        return global_seed if isinstance(global_seed, Iterable) else repeat(global_seed)
    elif rng is not None:

        def seeds():
            while True:
                yield rng.randint(0, 2**16 - 1)

        return seeds()
    else:
        raise ValueError(
            "Cannot generate sequence of global seeds when both 'rng' and 'global_seed' are None."
        )


def collate_data(
    worker_dir: str | Path, task_id: int, dest_dir: str | Path | None
) -> None:

    worker_dir = Path(worker_dir)  # ensure that worker_dir is a Path object
    dir_tree = DirectoryTree(worker_dir, task_id=task_id, config_id=-1, read_only=True)
    metric_reader = MetricReader(dir_tree)
    metric_reader.collate_all_data(dest_dir)


class DirectoryTree:
    """A class that translates a base directory into various relevant output directories.

    The following directory structure is used by DirectoryTree:

    <base_dir>
    |--> <task_id>/
    |    |--> dataset.parquet
    |    |--> metafeatures.parquet
    |    |--> labels/
    |    |    |--> <config_id>.parquet    <-- these are metric DataFrame files
    |    |--> configurations/
    |    |    |--> <config_id>.parquet    <-- these are metric DataFrame files
    |    |--> errors/
    |    |    |--> <config_id>.json    <-- these are error files
    """

    def __init__(
        self,
        base_dir: Path,
        task_id: int,
        config_id: int,
        read_only: bool = False,
    ):
        self.base_dir = base_dir.resolve()
        assert self.base_dir.exists() and self.base_dir.is_dir(), (
            f"The base directory and its parent directory tree must be created beforehand. Given base directory "
            f"either does not exist or is not a directory: {str(self.base_dir)}"
        )
        self.read_only = read_only
        self.task_id = task_id
        self.config_id = config_id

    @property
    def task_id(self) -> int | None:
        return self._taskid

    @task_id.setter
    def task_id(self, new_id: int | None):
        if new_id is not None:
            assert isinstance(
                new_id, int
            ), f"Task ID must be an integer, was given {type(new_id)}"
            self._taskid = new_id
            if self.read_only:
                return
            subdirs = [
                self.task_dir,
                self.labels_dir,
                self.configs_dir,
            ]
            for subdir in subdirs:
                if not subdir.exists():
                    subdir.mkdir(parents=True)
        else:
            self._taskid = None  # type: ignore

    @property
    def config_id(self) -> int | None:
        return self._config_id

    @config_id.setter
    def config_id(self, new_index: int | None):
        if new_index is not None:
            assert isinstance(
                new_index, int
            ), f"Model index must be an integer, was given {type(new_index)}"
            self._config_id = new_index
            if self.read_only:
                return
            subdirs = [self.configs_dir, self.labels_dir, self.errors_dir]
            for subdir in subdirs:
                if not subdir.exists():
                    subdir.mkdir(parents=True)
        else:
            self._config_id = None  # type: ignore

    @property
    def task_dir(self) -> Path:
        return self.base_dir / str(self.task_id)

    @property
    def dataset_file(self) -> Path:
        return self.task_dir / "dataset.parquet"

    @property
    def metafeatures_file(self) -> Path:
        return self.task_dir / "metafeatures.json"

    @property
    def labels_dir(self) -> Path:
        return self.task_dir / "labels"

    @property
    def configs_dir(self) -> Path:
        return self.task_dir / "configurations"

    @property
    def errors_dir(self) -> Path:
        assert (
            self.task_id is not None
        ), "A valid task_id needs to be set before the relevant model level subtree can be accessed."

        return self.task_dir / "errors"

    @property
    def labels_file(self) -> Path:
        assert (
            self.config_id is not None
        ), "A valid config_id needs to be set before the relevant file can be accessed."
        return self.labels_dir / f"{str(self.config_id)}.parquet"

    @property
    def configuration_file(self) -> Path:
        assert (
            self.config_id is not None
        ), "A valid config_id needs to be set before the relevant file can be accessed."
        return self.configs_dir / f"{str(self.config_id)}.parquet"

    @property
    def error_file(self) -> Path:
        assert (
            self.config_id is not None
        ), "A valid config_id needs to be set before the relevant file can be accessed."
        return self.errors_dir / f"{str(self.config_id)}.json"

    @property
    def collated_dir(self) -> Path:
        return self.task_dir / "collated"

    @property
    def existing_tasks(self) -> list[Path] | None:
        return (
            None
            if not self.base_dir.exists()
            else [d for d in self.base_dir.iterdir() if d.is_dir()]
        )

    @property
    def existing_configurations(self) -> list[Path] | None:
        return (
            None
            if not self.configs_dir.exists()
            else [d for d in self.configs_dir.iterdir() if d.is_file()]
        )


class MetricLogger:
    """
    The MetricLogger class handles the logging of metrics, configurations, datasets, and errors. It logs metrics into appropriate directories in Parquet format.

    Attributes:
        directory_tree (DirectoryTree): An instance of the DirectoryTree class that points to the correct directories.
        splits (list): A list containing the different types of data splits - "train", "valid", "test".
        logger (Logger): Logger for recording logs.
        _parquet_writer (pq.ParquetWriter): Writer for parquet files, initialized as None.

    Methods:
        close_parquet_writer: Closes the current ParquetWriter.
        log_dataset(dataset: Any): Logs the data from the provided dataset.
        log_configuration(config: dict): Logs a configuration by appending it to the existing configurations file.
        log_labels(split_name: str, y_proba: np.array, duration: float) -> None: Logs prediction labels and duration for a specific split.
        log_error(config, curr_global_seed, e, debug=False): Logs an error along with the current configuration and seed.
    """

    def __init__(self, directory_tree: DirectoryTree, logger: logging.Logger):
        """
        Initialize the MetricLogger with a directory structure and a logger.

        Parameters:
            directory_tree (DirectoryTree): An instance of the DirectoryTree class that
                points to the correct directories for logging various types of data.
            logger (logging.Logger): A configured Logger instance to log important events
                and debug information.

        Attributes:
            splits (List[str]): A list of splits to be used for data separation,
                defaults to ["train", "valid", "test"].
            _parquet_writer (Optional[pq.ParquetWriter]): An instance of a ParquetWriter
                for writing Parquet files. Initialized as None.
        """

        self.directory_tree = directory_tree
        self.logger = logger
        self.splits = ["train", "valid", "test"]
        self._parquet_writer = None

    def close_parquet_writer(self):
        """
        Close the ParquetWriter used by this instance of the MetricLogger.
        If no ParquetWriter is currently open, this method has no effect.

        This method should be used after all logging activities are completed
        to ensure that all resources are freed. After calling this method, the
        _parquet_writer attribute will be set to None.
        """
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

    def _write_dataframe_to_parquet(self, df: pd.DataFrame, filepath: Path | str):
        """
        Writes a given DataFrame to a Parquet file.

        Parameters:
            df (pd.DataFrame): DataFrame to write.
            filepath (str): Filepath to write to.
        """
        table = pa.Table.from_pandas(df)

        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(filepath, table.schema)

        assert self._parquet_writer is not None  # Assertion to satisfy the type checker
        self._parquet_writer.write_table(table)

    def log_dataset(self, dataset: Any):
        """
        Logs dataset splits.

        Parameters:
            dataset (Any): The dataset object with a method get_data() which returns a tuple (X, y).

        Raises:
            AttributeError: If the dataset object does not have a 'get_data' method.
        """

        if not hasattr(dataset, "get_data"):
            self.logger.error("Provided dataset object does not have a get_data method")
            raise AttributeError(
                "Provided dataset object does not have a get_data method"
            )

        if self.directory_tree.dataset_file.exists():
            self.logger.warning(
                f"Dataset file already exists: {self.directory_tree.dataset_file}. Skipping logging."
            )
            return

        metafeatures = MetaFeatures()

        for split in self.splits:
            X, y = dataset.get_data(split)

            df = pd.DataFrame(X)
            df["y"] = y
            df["split"] = split

            self._write_dataframe_to_parquet(df, self.directory_tree.dataset_file)

            metafeatures.update(df)

        # Write metafeatures to a JSON file
        with open(self.directory_tree.metafeatures_file, "w") as file:
            json.dump(metafeatures.to_dict(), file)

    def log_config(self, config: dict):
        """
        Logs a configuration by appending it to the existing configurations file.

        Parameters:
            config (dict): Configuration details in dictionary format.
        """

        config_df = pd.DataFrame.from_records([config])
        config_df.to_parquet(self.directory_tree.configuration_file)

    def log_labels(
        self,
        y_proba: np.ndarray,
        duration: float,
    ) -> None:
        """
        Save prediction information.

        Parameters:
            y_proba (np.array): Probability distribution of predictions.
            duration (float): Time duration for which predictions were made.
        """

        df = pd.DataFrame(
            y_proba, columns=[f"prediction_class_{i}" for i in range(y_proba.shape[1])]
        )
        df["duration"] = duration

        self._write_dataframe_to_parquet(df, self.directory_tree.labels_file)

    def log_error(
        self, config: Any, curr_global_seed: Any, e: Exception, debug: bool = False
    ):
        """
        Logs an error along with the current configuration and seed.

        Parameters:
            config (Any): Configuration details.
            curr_global_seed (Any): Current global seed.
            e (Exception): Exception to be logged.
            debug (bool): If set to True, raises the exception after logging. Default is False.

        Raises:
            e: Exception received as input.
        """

        if isinstance(config, Configuration):
            config = AttrDict(config.get_dictionary())

        error_description = {
            "exception": traceback.format_exc(),
            "config": config,
            "global_seed": curr_global_seed,
        }

        with open(
            self.directory_tree.errors_dir / f"{self.directory_tree.config_id}.json", "w"
        ) as fp:
            json.dump(error_description, fp, indent=4)

        if debug:
            raise e


class MetricReader:
    """
    The MetricReader class handles the reading of metrics, configurations, datasets, and labels.

    Attributes:
        directory_tree (DirectoryTree): An instance of the DirectoryTree class that points to the correct directories.
        logger (Logger): Logger for recording logs.
    """

    def __init__(
        self, directory_tree: DirectoryTree, logger: logging.Logger | None = None
    ):
        """
        Initialize the MetricReader with a directory structure and a logger.

        Parameters:
            directory_tree (DirectoryTree): An instance of the DirectoryTree class
                that points to the correct directories for reading various types of data.
            logger (logging.Logger): A configured Logger instance to log important
                events and debug information.
        """

        self.directory_tree = directory_tree

        if logger is None:
            logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        self.logger = logger

        # self.max_workers = 50

        self.config_files, self.labels_files = self._prepare_file_lists()

    def _prepare_file_lists(self):
        # Get the file names (without directory and extension)
        config_files = glob(os.path.join(self.directory_tree.configs_dir, "*.parquet"))
        labels_files = glob(os.path.join(self.directory_tree.labels_dir, "*.parquet"))

        # Get file names (without directory and extension)
        get_file_name = lambda x: os.path.splitext(os.path.basename(x))[0]

        config_names = {get_file_name(cf) for cf in config_files}
        labels_names = {get_file_name(lf) for lf in labels_files}

        # Filter the lists based on the intersection
        common_names = config_names & labels_names

        config_files = [cf for cf in config_files if get_file_name(cf) in common_names]
        labels_files = [lf for lf in labels_files if get_file_name(lf) in common_names]

        return config_files, labels_files

    @staticmethod
    def _read_single_parquet_file(
        file_path,
        drop_keys: list[str] | None = None,
        set_config_id_as_index=False,
        rename_cols=False,
    ):
        config_id = int(re.match(r".*?(\d+)\.parquet", Path(file_path).name).group(1))
        df = dd.read_parquet(file_path)

        if drop_keys is not None:
            df = df.drop(drop_keys, axis=1)

        if rename_cols:
            df = df.rename(
                columns={
                    col: f"{col}-{config_id}"
                    for col in df.columns
                    if col.startswith("prediction_class_") or col == "duration"
                }
            )

        # Set the index to the configuration ID
        if set_config_id_as_index:
            df.index = df.index.map(lambda _: config_id)

        return df

    def _read_from_parquet_files(
        self, files, drop_keys=None, set_config_id_as_index=False, rename_cols=False
    ):
        func = partial(
            self._read_single_parquet_file,
            drop_keys=drop_keys,
            set_config_id_as_index=set_config_id_as_index,
            rename_cols=rename_cols,
        )

        dataframes = []
        for file in tqdm(files, total=len(files)):
            df = func(file)
            dataframes.append(df)

        return dataframes

    @staticmethod
    def _set_divisions(df):
        if not df.known_divisions:
            # Reset index
            df_reset = df.reset_index()

            # Sort DataFrame in ascending order by index
            df_reset = df_reset.sort_values(by="index")

            # Compute and set index in Pandas
            df_pandas = df_reset.compute().set_index("index")

            # Convert back to Dask with from_pandas
            df = dd.from_pandas(df_pandas, npartitions=df.npartitions)
        return df

    def read_dataset(self):
        return self._set_divisions(dd.read_parquet(self.directory_tree.dataset_file))

    def read_labels(self):

        df = dd.concat(
            self._read_from_parquet_files(self.labels_files, rename_cols=True),
            axis=1,
        )

        df = df.repartition(npartitions=10)

        return self._set_divisions(df)

    def read_configs(self):
        # Load and concatenate DataFrames
        df = dd.concat(
            self._read_from_parquet_files(self.config_files, set_config_id_as_index=True)
        )

        # Repartition DataFrame
        df = df.repartition(npartitions=10)

        # Set divisions
        df = self._set_divisions(df)

        # Perform data transformations
        df = df.replace("none", None)
        df = df.categorize()
        df = dd.get_dummies(df, drop_first=True)

        return df

    def collate_all_data(self, dest_dir: str | Path | None = None):

        memory_usage_mb = lambda df: df.memory_usage(deep=True).sum().compute() / (
            1024**2
        )
        disk_usage_mb = lambda file: os.path.getsize(file) / (1024**2)

        process_data = [
            ("dataset", self.read_dataset),
            ("labels", self.read_labels),
            ("configs", self.read_configs),
        ]
        data_files = {
            name: self.directory_tree.collated_dir / f"{name}.parquet"
            for name, _ in process_data
        }

        for name, reader in process_data:
            self.logger.info(f"Processing {name}...")
            df = reader()
            self.logger.info(
                f"{name.capitalize()} processed. Size in memory: {memory_usage_mb(df):.2f} MB"
            )
            # TODO: make sure whether we overwrite or not
            df.to_parquet(data_files[name])
            self.logger.info(
                f"Compressed {name}. Size on disk: {disk_usage_mb(data_files[name]):.2f} MB"
            )
            with open(
                self.directory_tree.collated_dir / f"{name}_divisions.json", "w"
            ) as f:
                json.dump([int(x) for x in df.divisions], f)
            self.logger.info(f"Data written to {data_files[name]}")

        if dest_dir is None:
            archive_dir = self.directory_tree.collated_dir
        else:
            archive_dir = Path(dest_dir)  # Convert to Path object if not already
            shutil.copy(self.directory_tree.metafeatures_file, archive_dir)

        archive_dir = Path(archive_dir)
        archive_dir.mkdir(exist_ok=True)

        archive_file = archive_dir / f"collated_data-{self.directory_tree.task_id}"
        shutil.make_archive(
            str(archive_file), "zip", str(self.directory_tree.collated_dir)
        )

        self.logger.info(
            f"Final archive size: {disk_usage_mb(archive_file.with_suffix('.zip')):.2f} MB"
        )
        self.logger.info(f"Collated task {self.directory_tree.task_id}!!!")
        self.logger.info(f"Archive file: {archive_file.with_suffix('.zip')}")
