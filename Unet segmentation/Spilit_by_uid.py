#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按病例划分 8:2  →  train / val
并在各自子集中再次下采样负样本，保证 **负:正≈2:1**
"""
import csv, glob, random, cv2, numpy as np
from pathlib import Path, PurePath
from collections import defaultdict

IMG_DIR   = Path('Unet_dataset/images')
MASK_DIR  = Path('Unet_dataset/masks')
TRAIN_CSV = 'train_uid_list.csv'
VAL_CSV   = 'val_uid_list.csv'
VAL_RATIO      = 0.2        # 按病例划 20% 做验证
NEG_POS_RATIO  = 1.0        # 最终希望的 负:正
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# 1) group by seriesUID
sid2imgs = defaultdict(list)
for p in glob.glob(str(IMG_DIR/'*_img.png')):
    sid = PurePath(p).stem.split('_')[0]
    sid2imgs[sid].append(Path(p))

# 2) split by patient
uids = list(sid2imgs)
random.shuffle(uids)
split = int(len(uids)*(1-VAL_RATIO))
train_uids, val_uids = uids[:split], uids[split:]

def collect_rows(uids):
    rows = []
    for uid in uids:
        for img_p in sid2imgs[uid]:
            mask_p = MASK_DIR / img_p.name.replace('_img', '_mask')
            if mask_p.exists():
                rows.append([str(img_p), str(mask_p)])
    return rows

def downsample_neg(rows, ratio):
    pos = [r for r in rows if cv2.imread(r[1],0).max()>0]
    neg = [r for r in rows if cv2.imread(r[1],0).max()==0]
    keep_neg = random.sample(neg, min(len(neg), int(len(pos)*ratio)))
    rows = pos + keep_neg
    random.shuffle(rows)
    return rows, len(pos), len(keep_neg)

train_rows, n_pos_tr, n_neg_tr = downsample_neg(collect_rows(train_uids), NEG_POS_RATIO)
val_rows,   n_pos_val,n_neg_val= downsample_neg(collect_rows(val_uids),   NEG_POS_RATIO)

csv.writer(open(TRAIN_CSV,'w',newline='')).writerows(train_rows)
csv.writer(open(VAL_CSV,'w',newline='')).writerows(val_rows)

print('✅ Split done')
print(f'   train: {len(train_rows)} (pos {n_pos_tr} | neg {n_neg_tr})')
print(f'   val  : {len(val_rows)}  (pos {n_pos_val} | neg {n_neg_val})')
