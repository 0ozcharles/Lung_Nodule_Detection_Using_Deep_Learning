#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre‑process LUNA16 volumes for U‑Net training
--------------------------------------------
* 所有正样本必留
* 抽取负样本，使 **负:正 = RATIO_NEG_POS:1**
* 输出 PNG 到  Unet_dataset/images  &  Unet_dataset/masks
"""
import os, csv, glob, random
from pathlib import Path
import SimpleITK as sitk
import numpy as np, cv2
from tqdm import tqdm

# ========= 可调参数 =========
RAW_ROOT        = Path(r'D:\Luna16')
SAVE_ROOT       = Path('./Unet_dataset')
IMG_SIZE        = 320
RATIO_NEG_POS   = 3.0          # ← 想让负样本是正样本的 3 倍
SEED            = 42           # 固定随机种子，方便复现
# ===========================

random.seed(SEED)
np.random.seed(SEED)

IMG_DIR, MSK_DIR = SAVE_ROOT/'images', SAVE_ROOT/'masks'
IMG_DIR.mkdir(parents=True, exist_ok=True)
MSK_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 读标注 ----------
anno_path = RAW_ROOT/'annotations.csv'
series2annos = {}
with open(anno_path) as f:
    for row in csv.DictReader(f):
        sid = row['seriesuid']
        series2annos.setdefault(sid, []).append(
            [float(row['coordZ']), float(row['coordY']),
             float(row['coordX']), float(row['diameter_mm'])])

def world2voxel(world, origin, spacing):
    return np.abs(world - origin) / spacing   # xyz→zyx

pos_cnt, neg_cache = 0, []

# ---------- 主循环 ----------
SUBSETS = ['subset0', 'subset1', 'subset2']        # 想要的子集
mhd_files = []
for sub in SUBSETS:
    mhd_files += glob.glob(str(RAW_ROOT / sub / '*.mhd'))
for mhd in tqdm(sorted(mhd_files), desc='Volumes'):
    sid = Path(mhd).stem
    if sid not in series2annos:
        continue

    itk_img = sitk.ReadImage(mhd)
    vol     = sitk.GetArrayFromImage(itk_img).astype(np.int16)      # z,y,x
    origin  = np.array(itk_img.GetOrigin())[::-1]
    spacing = np.array(itk_img.GetSpacing())[::-1]

    vol = np.clip((vol + 1000) / 1400.0, 0, 1)                      # HU→[0,1]

    nodules = []
    for wz, wy, wx, diam in series2annos[sid]:
        vz, vy, vx = world2voxel(np.array([wz, wy, wx]), origin, spacing)
        nodules.append([vz, vy, vx, diam / spacing[0]])

    for iz, sli_hu in enumerate(vol):
        sli = cv2.resize((sli_hu*255).astype(np.uint8),
                         (IMG_SIZE, IMG_SIZE), cv2.INTER_CUBIC)

        msk = np.zeros_like(sli, np.uint8)
        for vz, vy, vx, vd in nodules:
            if abs(vz-iz) <= vd/2:
                cx = vx * IMG_SIZE / vol.shape[2]
                cy = vy * IMG_SIZE / vol.shape[1]
                r  = vd * IMG_SIZE / vol.shape[2]
                cv2.circle(msk, (int(cx), int(cy)), int(r)+5, 255, -1)

        pid = f'{sid}_{iz:04d}'
        if msk.max():                                                        # ⬅ 正样本
            cv2.imwrite(str(IMG_DIR/f'{pid}_img.png'),  sli)
            cv2.imwrite(str(MSK_DIR/f'{pid}_mask.png'), msk)
            pos_cnt += 1
        else:                                                                # ⬅ 负样本暂存
            neg_cache.append((sli, msk, pid))

# ---------- 二次抽负样本 ----------
need_neg = int(pos_cnt * RATIO_NEG_POS)
random.shuffle(neg_cache)
for sli, msk, pid in neg_cache[:need_neg]:
    cv2.imwrite(str(IMG_DIR/f'{pid}_img.png'),  sli)
    cv2.imwrite(str(MSK_DIR/f'{pid}_mask.png'), msk)
neg_cnt = need_neg

print(f'\n✅ 完成！正 {pos_cnt} | 负 {neg_cnt}  (≈{neg_cnt/pos_cnt:.1f}:1)')
print(f'   images → {IMG_DIR}\n   masks  → {MSK_DIR}')
