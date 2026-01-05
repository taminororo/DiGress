import os
import numpy as np
import pandas as pd

# データが置いてある場所
raw_dir = 'data/qm9/qm9_pyg/raw'

# フォルダがない場合は作成
os.makedirs(raw_dir, exist_ok=True)

print(f"Saving files to: {raw_dir}")

# QM9データセットの総数は約130,831個
n_samples = 130831 
all_indices = np.arange(n_samples)

# ランダムにシャッフル（混ぜる）
np.random.shuffle(all_indices)

# データの振り分け（訓練用:10万、検証用:1万、テスト用:残り）
train_size = 100000
val_size = 10000

train_idx = all_indices[:train_size]
val_idx = all_indices[train_size:train_size+val_size]
test_idx = all_indices[train_size+val_size:]

# CSVファイルとして保存（index_col=0 に対応した形式）
# ダミーの列 'id' を追加して保存します

pd.DataFrame({'mol_id': train_idx}, index=train_idx).to_csv(os.path.join(raw_dir, 'train.csv'))
pd.DataFrame({'mol_id': val_idx}, index=val_idx).to_csv(os.path.join(raw_dir, 'val.csv'))
pd.DataFrame({'mol_id': test_idx}, index=test_idx).to_csv(os.path.join(raw_dir, 'test.csv'))

print("成功！3つのリストファイルを作成しました。")
