echo "Activating conda env..."
. /mnt/data/optimal/dingyiming/miniconda3/etc/profile.d/conda.sh
# 验证自己的conda是否激活
which python
# 激活需要的环境
conda activate DAFormer

# echo "Prepared Datasets"
# python tools/convert_datasets/dsec.py data/DSEC_Semantic --nproc 8

echo "Begin training!"
cd /mnt/data/optimal/dingyiming/Codes/semi-supervised-uda

############################################################################################
# cityscapes ------->  DSEC
# python run_experiments.py --config configs/xxformer/cs2dsec_semi_xxformer.py
# python run_experiments.py --config configs/xxformer/cs2dsec_e2vid_offline_semi_xxformer.py
# python run_experiments.py --config configs/xxformer/cs2dsec_e2vid_online_semi_xxformer.py
# python run_experiments.py --config configs/xxformer/cs2dsec_e2vid_online_semi_dv3.py

############################################################################################
# GTA5       ------->  DSEC
# python run_experiments.py --config configs/xxformer/gta2dsec_semi_xxformer.py
# python run_experiments.py --config configs/xxformer/gta2dsec_e2vid_offline_semi_xxformer.py
python run_experiments.py --config configs/xxformer/gta2dsec_e2vid_online_semi_xxformer.py

############################################################################################
# cityscapes       ------->  ddd17
# python run_experiments.py --config configs/xxformer/cs2ddd17_e2vid_online_semi_xxformer.py
# python run_experiments.py --config configs/xxformer/cs2ddd17_e2vid_online_semi_dv3.py