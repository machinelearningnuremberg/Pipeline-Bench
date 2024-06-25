import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np


import pipeline_bench.lib.core.search_space
import pipeline_bench.lib.utils as util
from pipeline_bench.data import Dataset

from pipeline_bench.lib.core.procs import train



# pipeline_id = 3654
# task_id = 23
# global_seed = None
# local_seed = 333


def run_task(
    base_dir: Path,
    task_dir: Path,
    task_id: int,
    pipeline_id: int = 0,
    local_seed: Optional[int] = None,
    global_seed: Optional[int] = None,
    debug: bool = True,
    logger: logging.Logger = None,
):
    rng = np.random.RandomState(np.random.Philox(seed=local_seed, counter=task_id))
    global_seed_gen = util.default_global_seed_gen(rng, global_seed)
    dir_tree = util.DirectoryTree(
        base_dir=base_dir, task_id=task_id, config_id=pipeline_id
    )

    if logger is None:
        logger = logging.getLogger(__name__)

    metric_logger = util.MetricLogger(
        directory_tree=dir_tree,
        logger=logger,
    )

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    config, pipeline, curr_global_seed = pipeline_bench.lib.core.search_space.config_sampler(
        global_seed_gen=global_seed_gen, config_id=pipeline_id
    )
    logger.info("Sampled new pipeline.")
    logger.debug(f"{config}")
    metric_logger.log_config(config)
    metric_logger.close_parquet_writer()

    # Dataset loading
    dataset = Dataset(task_id, random_state=local_seed, worker_dir=task_dir)
    metric_logger.log_dataset(dataset)
    metric_logger.close_parquet_writer()

    logger.debug(f"Logged new sample for task {task_id}, pipeline {pipeline_id}.")
    # Actual model training and evaluation
    try:

        if dir_tree.error_file.exists():
            os.remove(dir_tree.error_file)

        util.set_seed(curr_global_seed)
        logger.info(f"Setting global seed to {curr_global_seed}.")
        
        train(
            pipeline=pipeline,
            dataset=dataset,
            metric_logger=metric_logger,
        )
    except Exception as e:
        logger.info("Pipeline Training failed.")
        metric_logger.log_error(config, curr_global_seed, e, debug=True)
    else:
        # Clean-up after nominal program execution
        logger.info("Sampled pipeline trained successfully.")
        
        metric_logger.log_pipeline(pipeline)
        del pipeline  # Release memory

    # Clean up memory before terminating task
    del metric_logger
    del dir_tree



