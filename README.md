This repository contains code and experimental artifacts accompanying the master’s thesis "When Sparsity Breaks Connectivity: Restoring Functional Paths in Neural Networks". 
The implementation is provided for reproducibility of the results and analysis presented in the thesis.

This project assumes UNIX compliant operating system.
1. to setup conda environment run ``./setup.sh``
2. in the root directory run ``cat ./data/datasets.json | python prepare_datasets.py`` to download all datasets
3. run experiments by invoking ``find ./data/{PATH} -type f | python ./batch_experiments.py``; alternatively you can cat and pipe specific experiments directly to ``main.py`` or run ``./run_experiments.sh``
4. generate plots from Appendix A by running ``python theory_validation.py`` (requires modifying script itself)
5. generate plots from Appendix B by running ``./generate_plots.sh``