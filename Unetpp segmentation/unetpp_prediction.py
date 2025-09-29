# predict_unetpp.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

UNETPP_WEIGHTS = r'F:\Pulmonary-Nodule-Detection-Based-on-Deep-Learning\U-net segmentation\model\again\unet++_39-0.64.hd5'
INPUT_DIR = './Unet_dataset_test/images'
MASK_SAVE_DIR = './demo_results_1/masks'
BOXED_SAVE_DIR = './demo_results_1/boxed'
THRESHOLD = 2  # for drawing boxes

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(BOXED_SAVE_DIR, exist_ok=True)

def prepare_image(img):
    """预处理：灰度图→归一化→添加通道"""
    img = img.astype(np.float32) / 255.0
    return img.reshape((320, 320, 1))

def get_centers(mask, threshold=THRESHOLD):
    """根据预测mask获取结节中心点"""
    from scipy import ndimage
    from skimage import morphology
    centers = []
    mask_bin = (mask >= threshold).astype(np.uint8)
    mask_dil = morphology.binary_dilation(mask_bin, morphology.disk(1))
    label_im, nb = ndimage.label(mask_dil)
    for i in range(1, nb + 1):
        blob = (label_im == i)
        cy, cx = ndimage.center_of_mass(blob)
        centers.append([int(round(cy)), int(round(cx))])
    return centers

def plot_one_box(img, coord, color=(0, 0, 255), thickness=2):
    c1 = (int(coord[0]), int(coord[1]))
    c2 = (int(coord[2]), int(coord[3]))
    return cv2.rectangle(img, c1, c2, color, thickness)

def main():
    print("Loading model...")
    model = load_model(UNETPP_WEIGHTS, compile=False)
    is_deep_supervision = isinstance(model.output, list)

    file_list = sorted(os.listdir(INPUT_DIR))
    print(f"Total images: {len(file_list)}")

    imgs = []
    for fname in file_list:
        img = cv2.imread(os.path.join(INPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
        imgs.append(prepare_image(img))
    imgs = np.stack(imgs, axis=0)  # shape: (N, 320, 320, 1)

    print("Predicting...")
    preds = model.predict(imgs, batch_size=1)
    if is_deep_supervision:
        preds = preds[-1]  # 取 out4

    for i, pred in enumerate(preds):
        pred_mask = (pred.squeeze() * 255.).astype(np.uint8)  # shape: (320,320)
        save_name = file_list[i]

        # 保存mask
        cv2.imwrite(os.path.join(MASK_SAVE_DIR, save_name), pred_mask)

        # 画框
        centers = get_centers(pred_mask)
        ori_img = cv2.imread(os.path.join(INPUT_DIR, save_name))
        for y, x in centers:
            box = [x - 5, y - 5, x + 5, y + 5]
            ori_img = plot_one_box(ori_img, box)
        cv2.imwrite(os.path.join(BOXED_SAVE_DIR, save_name), ori_img)

    print("Prediction & visualization complete.")

if __name__ == '__main__':
    main()
