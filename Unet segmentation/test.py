import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
import os


THRESHOLD = 2  # 分割图的阈值

def unet_candidate_single(mask_path):
    centers = []
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 阈值 → 二值图
    mask_bin = (mask_img >= THRESHOLD).astype(np.uint8)

    # 膨胀
    mask_dilated = morphology.binary_dilation(mask_bin, footprint=morphology.disk(1))

    label_im, nb = ndimage.label(mask_dilated)
    for i in range(1, nb + 1):
        blob = (label_im == i)
        cy, cx = center_of_mass(blob)
        centers.append([int(round(cy)), int(round(cx))])
    return centers


def plot_one_box(img, coord, label=None, line_thickness=None):
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))
    color = [0, 0, 255]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    img = cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img


def process_single_image(img_path, mask_path, save_path):
    img = cv2.imread(img_path)
    centers = unet_candidate_single(mask_path)

    for cy, cx in centers:
        box = [cx - 5.5, cy - 5.5, cx + 5.5, cy + 5.5]
        img = plot_one_box(img, box)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"✅ 已保存: {save_path}")


if __name__ == "__main__":
    # 替换为你自己的图片路径
    image_path = r'F:\Pulmonary-Nodule-Detection-Based-on-Deep-Learning\U-net segmentation\Unet_dataset_test\images\1.3.6.1.4.1.14519.5.2.1.6279.6001.126631670596873065041988320084_0223_img.png'
    mask_path = r'F:\Pulmonary-Nodule-Detection-Based-on-Deep-Learning\U-net segmentation\Unet_dataset_test\masks\1.3.6.1.4.1.14519.5.2.1.6279.6001.126631670596873065041988320084_0223_mask.png'
    save_boxed_path = './boxed.png'

    process_single_image(image_path, mask_path, save_boxed_path)
