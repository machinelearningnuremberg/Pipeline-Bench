from __future__ import annotations

import logging
from enum import Enum, unique
from pathlib import Path

import numpy as np
import json
import zipfile
import pandas as pd
import dask.dataframe as dd


from pipeline_bench.utils import (
    ensure_ensembles_list_of_lists,
    ensure_datapoints_list_of_lists,
    get_column_names,
)
from pipeline_bench.lib.utils import AttrDict

try:
    from pipeline_bench.lib.core.sample import run_task
    from pipeline_bench.lib.utils import collate_data
except ImportError as e:
    DATA_CREATION_AVAILABLE = False
else:
    DATA_CREATION_AVAILABLE = True

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)


@unique
class BenchmarkTypes(Enum):
    Live = "live"
    Table = "table"
    Surrogate = "surrogate"  # TODO: implement one day


class Benchmark:
    _call_fn = None
    _dataset = pd.DataFrame()
    _configs = pd.DataFrame()
    _labels = pd.DataFrame()
    _metafeatures = AttrDict()
    _surrogates = None

    def __init__(
        self,
        task_id: int,
        mode: str | BenchmarkTypes = BenchmarkTypes.Live,
        # download: bool = False,
        worker_dir: str | Path | None = None,
        backend: str = "pandas",
        data_version: str = "micro",
        lazy: bool = False,  # pylint: disable=unused-argument
    ):
        if isinstance(mode, str):
            try:
                mode = BenchmarkTypes(mode.lower())
            except ValueError as e:
                raise ValueError(
                    f"Unknown mode: {mode}. Must be one of: {[x.value for x in BenchmarkTypes]}"
                ) from e

        self.task_id = task_id
        self.mode = mode
        self.data_version = data_version
        self.splits = ["train", "valid", "test"]

        # Directories
        worker_dir = Path(worker_dir) if worker_dir is not None else Path.cwd()

        self.base_dir = worker_dir / "pipeline_bench"
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.task_dir = worker_dir / "pipeline_bench_data" / str(task_id)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir = worker_dir / "pipeline_bench_tables" / str(task_id) / self.data_version
        self.table_dir.mkdir(parents=True, exist_ok=True)

        self.backend = backend
        # if download:
        #     if mode is BenchmarkTypes.Live:
        #         self.task_dir.mkdir(parents=True, exist_ok=True)
        #     if mode is BenchmarkTypes.Table:
        #         if not self.table_dir.exists():
        #             raise NotImplementedError(
        #                 "Table download is not implemented yet. Please ask Maciej for the file."
        #             )
        #     if mode is BenchmarkTypes.Surrogate:
        #         pass

        loaders = {
            BenchmarkTypes.Live: self._load_live,
            BenchmarkTypes.Table: self._load_table,
            BenchmarkTypes.Surrogate: self._load_surrogate,
        }

        loaders[mode]()

    def __call__(self, *args, **kwargs):
        if self._call_fn is None:
            raise RuntimeError("No benchmarking function defined. This is a bug.")

        return self._call_fn(*args, **kwargs)

    def _load_live(self):
        self._call_fn = self._benchmark_live

    def _benchmark_live(
        self,
        pipeline_id: int = 0,
        **kwargs,
    ) -> None:

        if not DATA_CREATION_AVAILABLE:
            raise RuntimeError(
                "Cannot create data without additional dependencies."
                " Please install pipeline_bench[create_data]"
            )

        run_task(
            base_dir=self.base_dir,
            task_dir=self.task_dir,
            task_id=self.task_id,
            pipeline_id=pipeline_id,
            local_seed=kwargs.get(
                "local_seed", 333
            ),  # If not provided in kwargs, default value will be used
            global_seed=kwargs.get("global_seed", None),
            debug=kwargs.get("debug", True),
            logger=kwargs.get("logger", None),
        )

    def _load_surrogate(self):
        pass

    def _load_table(self, table_names=["dataset", "labels", "configs"]):

        assert self.table_dir.exists() and self.table_dir.is_dir()
        zip_file = self.table_dir / f"{self.data_version}-{self.task_id}.zip"
        assert zip_file.exists() and zip_file.is_file()

        # Check if all the table_names.parquet files exist
        if not all((self.table_dir / f"{table_name}.parquet").exists() for table_name in table_names):
            with zipfile.ZipFile(zip_file, "r") as archive:
                archive.extractall(self.table_dir)

        # Load divisions and DataFrames
        for table_name in table_names:
            if self.backend == "dask":
                with open(self.table_dir / f"{table_name}_divisions.json") as f:
                    divisions = json.load(f)
                divisions.sort()
                dataframe = dd.read_parquet(self.table_dir / f"{table_name}.parquet")
                dataframe.divisions = tuple(divisions)
            elif self.backend == "pandas":
                dataframe = pd.read_parquet(self.table_dir / f"{table_name}.parquet")
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            setattr(self, f"_{table_name}", dataframe)

        # Load metafeatures from JSON
        with open(self.table_dir / "metafeatures.json") as file:
            self._metafeatures = AttrDict(json.load(file))

        self._call_fn = self._benchmark_table

    @ensure_ensembles_list_of_lists
    @ensure_datapoints_list_of_lists
    def _benchmark_table(
        self,
        ensembles: list[list[int]],
        datapoints: list[np.ndarray],
        get_probabilities: bool = False,
        aggregate: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """
        Function to benchmark a given set of ensembles on provided datapoints.

        Parameters:
            ensembles (list[list[int]]): List of ensembles, where each ensemble is represented as a list of pipeline IDs.
            datapoints (list[np.ndarray]): List of datapoints to evaluate the ensembles on.
            get_probabilities (bool): If True, the function will return probabilities of class predictions. If False, it will return class predictions.
            aggregate (bool): If True, it averages predictions across pipelines in an ensemble.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The benchmark results in a 4D numpy array of shape (B, D, N, C) where:
                - B is the number of ensembles,
                - D is the number of datapoints,
                - N is the number of pipelines in the ensemble.
                - C is the number of classes,

        Raises:
            ValueError: If a value error occurs during the benchmarking process.
        """

        # Flatten the list of lists to get all pipeline IDs
        pipeline_ids = [str(pid) for ensemble in ensembles for pid in ensemble]

        # Get corresponding column names
        columns = get_column_names(
            pipeline_ids, num_classes=self._metafeatures.num_classes
        )

        # Flatten the list of lists to get all datapoint IDs
        datapoint_ids = [dp for sublist in datapoints for dp in sublist]

        # Filter the dataframe for the columns and rows corresponding to the pipeline_ids and datapoints
        df = self._labels.loc[datapoint_ids, columns]

        if self.backend == "dask":
            # Compute only the selected rows
            df = df.compute()

        # Check if the dataframe is empty, which would mean no match was found
        if df.empty:
            raise ValueError(f"No row with pipeline_id in {pipeline_ids} found.")

        # Convert DataFrame to numpy array
        array = df.values

        # Split the array into separate arrays for each ensemble
        ensemble_arrays = np.split(array, len(ensembles), axis=1)

        # Determine the maximum number of pipelines in an ensemble
        max_pipelines = max(len(ensemble) for ensemble in ensembles)

        # Reshape each ensemble array separately and then stack them along a new dimension
        reshaped_arrays = [
            ensemble_array.reshape(
                (
                    len(datapoints[0]),
                    max_pipelines,
                    self._metafeatures.num_classes,
                )
            )
            for ensemble_array in ensemble_arrays
        ]

        # Stack the reshaped arrays along a new dimension
        array = np.stack(reshaped_arrays, axis=0)

        # If probabilities are not required, get class predictions by choosing the class with the highest probability
        if not get_probabilities:
            array = np.argmax(array, axis=-1, keepdims=True)

        # If aggregation is required, average predictions across pipelines in an ensemble
        if aggregate:
            array = np.mean(array, axis=-2, keepdims=True)

        return array

    @staticmethod
    def sample_config(random_state: int | np.random.RandomState | None | None = None):

        from pipeline_bench.lib.core.search_space import get_search_space

        search_space = get_search_space()
        search_space.seed(random_state)
        config = search_space.sample_configuration()

        return config

    def get_splits(self, return_train: bool = False, return_array: bool = False) -> dict:
        """Create the split datasets."""

        results = {}

        splits = self.splits.copy()
        # pylint: disable=expression-not-assigned
        splits.remove("train") if not return_train and "train" in splits else None

        for split in splits:
            split_data = self._dataset.loc[self._dataset["split"] == split]
            if self.backend == "dask":
                split_data = split_data.compute()

            if split_data.empty:
                raise ValueError(f"No data found for split '{split}'.")

            if return_array:
                results[f"X_{split}"] = split_data.drop(columns=["split", "y"]).values
                results[f"y_{split}"] = split_data["y"].values
            else:
                results[f"X_{split}"] = split_data.drop(
                    columns=["split", "y"]
                ).index.values
                results[f"y_{split}"] = split_data["y"].index.values

        return results

    @ensure_ensembles_list_of_lists
    def get_pipeline_features(self, ensembles: list[list[int]]) -> np.ndarray:
        """
        Extract the features corresponding to each pipeline in the provided ensembles.

        Parameters:
            ensembles (list[list[int]]): List of ensembles. Each ensemble is a list of pipeline IDs.

        Returns:
            np.ndarray: A 3D numpy array with shape (B, N, F), where:
                    - B is the number of ensembles,
                    - N is the maximum number of pipelines in any ensemble, and
                    - F is the number of features per pipeline.

        Raises:
            ValueError: If a value error occurs during the benchmarking process.
        """

        # Flatten the list of ensembles into a single list of pipeline IDs
        pipeline_ids = [pid for ensemble in ensembles for pid in ensemble]

        # Filter the dataframe to include only the pipelines in the provided list
        df = self._configs.loc[self._configs.index.isin(pipeline_ids)]

        if self.backend == "dask":
            # Compute only the selected rows
            df = df.compute()

        # Repeat rows in dataframe to match the count in the original pipeline_ids list
        df = df.loc[df.index.repeat([pipeline_ids.count(x) for x in df.index])]

        # Check if the dataframe is empty, raise ValueError if so
        if df.empty:
            raise ValueError(f"No row with pipeline_id in {pipeline_ids} found.")

        # Convert DataFrame to numpy array
        array = df.values

        # Determine the maximum number of pipelines in an ensemble
        max_pipelines = max(len(ensemble) for ensemble in ensembles)

        # Reshape the array to match the shape of the input ensembles list
        array = array.reshape((len(ensembles), max_pipelines, df.shape[1]))

        return array

    def get_metafeatures(self) -> np.ndarray:

        metafeatures = [
            "num_datapoints",
            "num_classes",
            "num_categorical",
            "num_numerical",
        ]

        return np.array([self._metafeatures[mf] for mf in metafeatures])

    def get_hp_candidates_ids(self) -> list:

        return list(range(0, self._metafeatures["num_pipelines"]))

    def collate_data(self) -> None:

        if not DATA_CREATION_AVAILABLE:
            raise RuntimeError(
                "Cannot create data without additional dependencies."
                " Please install pipeline_bench[create_data]"
            )

        collate_data(
            worker_dir=self.base_dir, task_id=self.task_id, dest_dir=self.table_dir,
            data_version=self.data_version
        )

        self._load_table()
