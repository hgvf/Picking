# Picking

### Related files
* [Experimental Results](https://docs.google.com/spreadsheets/d/1GNUUGtq8gjdsK_Qu6LqYDZW5WN3AU1dzf668a1C6SDk/edit#gid=0)

---
## File structures
 
    ├── Basic scripts                 # 基本訓練模型都會用到的
    │   ├── train.py                  # Training
    │   ├── find_threshold.py         # 把訓練好的模型拿來測試，並找出最佳 picking rule 
    │   ├── model.py                  # Picking models
    │   ├── plot.py                   # 把訓練好的模型拿來實際預測並畫圖
    │   ├── whole_sample_test.py      # 把資料完整長度保留，依此直接做測試
    │   ├── utils.py                  # 訓練時會用到的重要 functions
    │   ├── optimizer.py              # Noam optimizer
    │   ├── gen_stat.py               # 把 find_threshold.py 產生的檔案，畫出相關測試統計圖表
    │   ├── wmseg_dataparallel.py     # 多 GPU 時支援
    │
    ├── Trace characteristic    # 計算每筆 trace 數據的
    │   ├── snr.py              # 計算 SNR
    │   ├── calc.py             # 計算新制震度
    │
    ├── Pretrained embedding scripts    # 使用 time-series pretrain embedding 會用到
    │   ├── emb_train.py                # Training
    │   ├── emb_model.py                # Pretrained model
    │   ├── emb_loss.py                 # Pretrained loss functions
    │   ├── modules.py                  # Pretrained model related modules
    │   
    ├── TemporalSegmentation    # Temporal Segmentation modules
    │   ├── ...
    │   
    ├── conformer                   # conformer model
    │   ├── ...
    │   
    └── README.md
