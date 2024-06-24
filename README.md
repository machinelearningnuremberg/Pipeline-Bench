# Pipeline-Bench

## Exploring the Search Space

To fully explore the pipeline parameterization, use the following Python code:

```python
from pipeline_bench.lib.core.search_space import get_search_space
from pprint import pprint

# Retrieve the search space
search_space = get_search_space()

# Print the search space
pprint(search_space)

# Print the names of the hyperparameters
hps = search_space.get_hyperparameter_names()
pprint(hps)
```

The search space consists of several key aspects, some of which are highlighted below:

### Classifiers

- adaboost
- bernoulli_nb
- decision_tree
- extra_trees
- gaussian_nb
- gradient_boosting
- k_nearest_neighbors
- lda
- liblinear_svc
- libsvm_svc
- mlp
- multinomial_nb
- passive_aggressive
- qda
- random_forest
- sgd

### Feature Preprocessors

- extra_trees_preproc_for_classification
- fast_ica
- feature_agglomeration
- kernel_pca
- kitchen_sinks
- liblinear_svc_preprocessor
- no_preprocessing
- nystroem_sampler
- pca
- polynomial
- random_trees_embedding
- select_percentile_classification
- select_rates_classification

### Categorical Encoding

- encoding
- no_encoding
- one_hot_encoding

### Rescaling

- minmax
- none
- normalize
- power_transformer
- quantile_transformer
- robust_scaler
- standardize

## Contributing to Pipeline-Bench

If you'd like to contribute to Pipeline-Bench, follow the guidelines below:

### Managing git submodules

For working with submodules, refer to the [git-scm documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules). You can pull changes for `Pipeline-Bench` and all its submodules (`auto-sklearn`) using the command:

```bash
git pull --recurse-submodules
```

### Optional: Install miniconda and create an environment

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/.conda  # Change to place of preference
rm install_miniconda.sh
```

Consider running `~/.conda/bin/conda init` or `~/.conda/bin/conda init zsh`.

Create the environment and activate it

```
conda create -n Pipeline-Bench python=3.10
conda activate Pipeline-Bench
```

### Install poetry

First, [install poetry](https://python-poetry.org/docs), e.g., via

```
curl -sSL https://install.python-poetry.org | python3 -
```

Consider appending `export PATH="$HOME/.local/bin:$PATH"` into `~/.zshrc` / `~/.bashrc`.

### Let poetry take care of all dependencies

```
poetry install
```

In case you do do not wish to create any data (use "live" API), run

```
poetry install --extras "without_data_creation"
```

To install a new dependency use `poetry add dependency` and commit the updated `pyproject.toml` to git.

### Activate pre-commit

```
pre-commit install
```

Consider appending `--no-verify` to your urgent commits to disable checks.

### Working with git submodules

See [the git-scm documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules). In short:

To pull in changes for `Pipeline-Bench` and all submodules (`auto-sklearn`) run

```bash
git pull --recurse-submodules
```

<!-- ### Working with OpenML

To use the OpenML API, you need to create an account on [OpenML](https://www.openml.org/). Then, create a file `pipeline_bench/data/openml_config` with the following content:

```
[openml]
apikey = <your api key>
``` -->
