import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
    os.system("python train.py \
                --data_dir ./dataset \
                --name enlightening \
                --patchD \
                --patch_vgg \
                --n_patch 5 \
                --fineSize 320 \
                --patchSize 32 \
                --batchSize 32 \
                --use_wgan 0 \
                --use_ragan \
                --hybrid_loss \
                --vgg 1 \
                --vgg_choose relu5_1")
elif opt.predict:
    for i in range(1):
        os.system("python predict.py \
                    --data_dir ./dataset \
                    --name enlightening \
                    --use wgan 0 \
                    --which_epoch" + str(200 - i * 5))