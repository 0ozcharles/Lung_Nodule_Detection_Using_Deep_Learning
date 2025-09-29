import os, glob, numpy as np, cv2
from medpy.metric import binary

GT_DIR   = r"./Unet_dataset_test/masks/"
PRED_DIR = r"./demo_results/masks/"
VOXEL_SPACING = (1.0, 1.0)

dice_all, jac_all, hd95_all = [], [], []

for gt_path in glob.glob(os.path.join(GT_DIR, "*_mask.png")):
    pred_path = os.path.join(
        PRED_DIR,
        os.path.basename(gt_path).replace("_mask.png", "_img.png")
    )
    if not os.path.exists(pred_path):
        print("missing:", pred_path); continue

    gt_img   = cv2.imread(gt_path,  0)
    pred_img = cv2.imread(pred_path, 0)

    # ----------- 自动判断前景灰度 -----------
    vals, counts = np.unique(gt_img, return_counts=True)
    if len(vals) != 2:
        print("非二值图:", gt_path); continue
    fg_val = vals[np.argmin(counts)]   # 像素更少者为前景
    gt   = (gt_img   == fg_val)
    pred = (pred_img == fg_val)

    # ----------- 计算指标 -----------
    dice_all.append(binary.dc(pred, gt) * 100)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or (pred, gt).sum()
    jac_all.append(inter / union * 100 if union else np.nan)

    if gt.any() and pred.any():
        hd95_all.append(binary.hd95(pred, gt, voxelspacing=VOXEL_SPACING))
    else:
        hd95_all.append(np.nan)

# ----------- 输出 -----------
print(f"Dice   : {np.nanmean(dice_all):.2f} %")
print(f"Jaccard: {np.nanmean(jac_all):.2f} %")
dice_frac = np.nanmean(dice_all) / 100
print(f"(理论 J≈{dice_frac/(2-dice_frac)*100:.2f} %)")
print(f"HD95mm : {np.nanmean(hd95_all):.2f} "
      f"(有效 {np.sum(~np.isnan(hd95_all))}/{len(hd95_all)} slices)")
