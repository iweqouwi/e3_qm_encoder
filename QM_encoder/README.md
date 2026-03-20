nohup python -u src/train_qmmm.py --config configs/train_qmmm_config.yaml > ../train_qmmm.log 2>&1 &
[1] 1296847
如果需要加载训练参数：
nohup python -u src/train_qmmm.py --config configs/train_qmmm_config.yaml --checkpoint /home/hk/project/QM_encoder/results/qmmm_encoder/checkpoint.pt > ../train_qmmm.log 2>&1 &
