#!/bin/bash

# Define base directory for pipeline_bench
BASE_DIR="pipeline_bench/lib"

# Define the paths for tabrepo and autogluon-bench
TABREPO_DIR="$BASE_DIR/tabrepo"

# Clone and install tabrepo if it doesn't exist
if [ ! -d "$TABREPO_DIR/tabrepo" ]; then
    git clone https://github.com/autogluon/tabrepo.git "$TABREPO_DIR/tabrepo"
    pip install -e "$TABREPO_DIR/tabrepo"
fi

# Clone and install autogluon-bench if it doesn't exist
if [ ! -d "$TABREPO_DIR/autogluon-bench" ]; then
    git clone https://github.com/autogluon/autogluon-bench.git "$TABREPO_DIR/autogluon-bench"
    # Check if setup.py or pyproject.toml exists before installing
    if [ -f "$TABREPO_DIR/autogluon-bench/setup.py" ] || [ -f "$TABREPO_DIR/autogluon-bench/pyproject.toml" ]; then
        pip install -e "$TABREPO_DIR/autogluon-bench"
    else
        echo "ERROR: '$TABREPO_DIR/autogluon-bench' cannot be installed. No 'setup.py' or 'pyproject.toml' found."
    fi
else
    echo "Autogluon-bench already installed."
fi

echo "Installation completed."
