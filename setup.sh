# in some environments you may need to copy paste it straight into your terminal
conda create --name gt -c conda-forge python=3.12 graph-tool
conda activate gt
conda install numpy scipy matplotlib statsmodels imageio tqdm scikit-learn
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia