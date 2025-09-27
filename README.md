1. to download dependencies run ``./setup.sh``
2. in the root directory run ``cat ./data/datasets.json | python prepare_datasets.py`` to download all datasets
2. run experiments by invoking ``find ./data/baselines -type f | python ./batch_experiments.py`` and ``find ./data/population_test -type f | python ./batch_experiments.py``
3. generate all plots by running ``bash ./generate_all_plots.sh`` 