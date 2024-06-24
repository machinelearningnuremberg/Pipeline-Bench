from __future__ import annotations

import time

import pipeline_bench.lib.utils as util


from pipeline_bench.lib.auto_sklearn.classification_pipeline import SimpleClassificationPipeline
from sklearn.utils.validation import check_is_fitted

import sklearn

from pipeline_bench.data import Dataset


def _main_proc(
    pipeline: SimpleClassificationPipeline,
    dataset: Dataset,
    metric_logger: util.MetricLogger,
    split_name: str = "train",
):

    inputs, targets = dataset.get_data(split_name=split_name)

    start_time = time.time()
    if split_name == "train":
        breakpoint()
        pipeline = pipeline.fit(inputs, targets)
    y_proba = pipeline.predict_proba(inputs.copy())
    end_time = time.time()

    duration = (end_time - start_time) / len(inputs)
    metric_logger.log_labels(y_proba=y_proba, duration=duration)

    diverged = False
    try:
        for _, step in pipeline.named_steps.items():
            check_is_fitted(step)
    except sklearn.exceptions.NotFittedError:
        metric_logger.logger.debug(
            f"config={pipeline} raised NotFittedError unexpectedly!"
        )
        diverged = True

    if diverged:
        return True

    return False


def train(
    pipeline: SimpleClassificationPipeline,
    dataset: Dataset,
    metric_logger: util.MetricLogger,
):

    for split in ["train", "valid", "test"]:

        diverged = _main_proc(
            pipeline=pipeline,
            dataset=dataset,
            split_name=split,
            metric_logger=metric_logger,
        )

    metric_logger.close_parquet_writer()

    if diverged:
        raise RuntimeError("Pipeline has not been fitted!!!")
