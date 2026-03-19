/project/QM_encoder$ nohup python -u src/train_qmmm.py --config configs/train_qmmm_config.yaml > ../train_qmmm.log 2>&1 &
[1] 1296847
如果需要加载训练参数：
python src/train_qmmm.py --config configs/train_qmmm_config.yaml --checkpoint results/.../checkpoint.pt
