# seeds generated using: python -c "import random; print(random.randint(1, 2**32))"
set -euo pipefail

# set 1
find ./data/baselines/set_1/custom/mag_uni/dst/erk/99875/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1456423536 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/999/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=4275656331 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/9975/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=3324325736 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/99/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2709818336 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/995/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=3090730487 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=170584870 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/995/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=3052529488 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/9975/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=3846713997 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/999/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=142615278 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99875/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=95808356 --runs=5

find ./data/baselines/set_1/custom/mag_uni/dst/erk/99875/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1976825178 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/999/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1970446711 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/9975/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=335813560 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/99/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=2165849086 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/995/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1481596295 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99875/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3218377658 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/999/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1219932179 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/9975/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=748619615 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=2271654420 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/995/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=549286591 --runs=5

find ./data/baselines/set_1/custom/mag_uni/dst/erk/99875/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1428445849 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/999/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=864907760 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/9975/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1408823224 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/99/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1041091603 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/erk/995/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3669503381 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99875/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=4179287160 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/999/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=2147852766 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/9975/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=216446170 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/99/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1159583697 --runs=5
find ./data/baselines/set_1/custom/mag_uni/dst/uniform/995/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3062872710 --runs=5

# set 2
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99875/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2783408643 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/999/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1989336581 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/9975/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1869864854 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2247238965 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/995/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2021939731 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=728231654 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/995/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1577731556 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/9975/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2931847635 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/999/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1675086887 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99875/tiny_imagenet -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=3530351732 --runs=5

find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99875/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=4048381046 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/999/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1771689505 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/9975/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=470486275 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=67864967 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/995/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1636341810 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99875/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=18039801 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/999/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1422878781 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/9975/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3679749717 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1084852892 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/995/cifar10 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=672983674 --runs=5

find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99875/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=2012254306 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/999/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=517323282 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/9975/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3173973642 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/99/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1641114359 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/erk/995/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=153012461 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99875/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=4201705765 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/999/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=3008633460 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/9975/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1112732635 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/99/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1325584729 --runs=5
find ./data/baselines/set_2/custom/gmag_guni/dst/uniform/995/cifar100 -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=14 --seed=1091550534 --runs=5

# resnet34
find ./data/baselines/set_2/resnet34/zero/dst/gmag_guni -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2634444911 --runs=5
find ./data/baselines/set_2/resnet34/zero/dst/mag_grad -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=103818178 --runs=5
find ./data/baselines/set_2/resnet34/zero/dst/mag_uni -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=1071005823 --runs=5
find ./data/baselines/set_2/resnet34/zero/dst/static_sparsity -type f | LC_ALL=C sort | python ./batch_experiments.py --workers=7 --seed=2950005666 --runs=5
