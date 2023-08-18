import numpy as np


def ensure_ensembles_list_of_lists(func):
    def wrapper(
        self,
        *args,
        **kwargs,
    ):
        ensembles = kwargs.get("ensembles")
        # If single int, convert it to a list of lists
        if isinstance(ensembles, int):
            ensembles = [[ensembles]]
        # If the list of int, convert it to a list of lists
        elif isinstance(ensembles, list) and all(isinstance(i, int) for i in ensembles):
            ensembles = [ensembles]
        # If it's a numpy array, convert it to a list of lists
        elif isinstance(ensembles, np.ndarray):
            ensembles = ensembles.tolist()
        # If not a list of lists at this point, raise an error
        elif not (
            isinstance(ensembles, list) and all(isinstance(i, list) for i in ensembles)
        ):
            raise ValueError(
                "Ensembles input should be an int, list[int], numpy array, or list[list[int]]"
            )
        kwargs["ensembles"] = ensembles
        return func(self, *args, **kwargs)

    return wrapper


def ensure_datapoints_list_of_lists(func):
    def wrapper(
        self,
        *args,
        **kwargs,
    ):
        datapoints = kwargs.get("datapoints")
        # If datapoint is a single numpy.int64, convert it to a list of numpy arrays
        if isinstance(datapoints, np.int64):
            datapoints = [np.array([datapoints])]
        # If datapoints is a list of numpy.int64, convert it to a list of numpy arrays
        elif isinstance(datapoints, list) and all(
            isinstance(i, np.int64) for i in datapoints
        ):
            datapoints = [np.array(datapoints)]
        # If it's a numpy array, convert it to a list of lists
        elif isinstance(datapoints, np.ndarray):
            datapoints = [datapoints.tolist()]
        # If not a list of lists at this point, raise an error
        elif not (
            isinstance(datapoints, list)
            and all(isinstance(i, list) or isinstance(i, np.ndarray) for i in datapoints)
        ):
            raise ValueError(
                "Datapoints input should be a numpy.int64, list[numpy.int64], numpy array, or list[list[numpy.int64]]"
            )
        kwargs["datapoints"] = datapoints
        return func(self, *args, **kwargs)

    return wrapper


def get_column_names(pipeline_ids: list[str], num_classes: int) -> list[str]:
    """
    Convert pipeline_ids to corresponding column names in the dataframe.

    Arguments:
    pipeline_ids -- a list of pipeline indices

    Returns:
    List of column names corresponding to the pipelines
    """
    # Get corresponding column names for predictions only
    column_names = []
    for pid in pipeline_ids:
        for i in range(num_classes):  # assuming _num_classes is the number of classes
            column_names.append(f"prediction_class_{i}-{pid}")
    return column_names
