# 模型预测的相关功能

from Train_Unet import get_unet
import glob
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
import os

CHANNEL_COUNT = 1
UNET_WEIGHTS = r'F:\Pulmonary-Nodule-Detection-Based-on-Deep-Learning\U-net segmentation\model\again\unet++_39-0.64.hd5'
THRESHOLD = 2
BATCH_SIZE = 1
CUBE_SIZE = 32  # 预测模型所处理的三维图像大小：32x32x32

# 获取unet预测结果的中心点坐标(x,y)
def unet_candidate_dicom(unet_result_path):
    centers  = []
    image_t  = cv2.imread(unet_result_path, cv2.IMREAD_GRAYSCALE)

    # 阈值化 → bool
    mask = (image_t >= THRESHOLD).astype(np.uint8)

    # 形态学膨胀
    footprint   = morphology.disk(1)          # ← 旧 selem
    image_dil   = morphology.binary_dilation(mask, footprint=footprint)

    label_im, nb = ndimage.label(image_dil)
    for i in range(1, nb + 1):
        blob   = (label_im == i)
        cy, cx = center_of_mass(blob)
        centers.append([int(round(cy)), int(round(cx))])
    return centers



# 数据输入网络之前先进行预处理
def prepare_image_for_net(img):
    img = img.astype(np.float32)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        img = img.reshape(1, img.shape[-2], img.shape[-1], 1)
    return img

from tensorflow.keras.models import load_model
# unet模型的预测代码
def unet_predict(imagepath, maskpath):
    # model = get_unet()
    # model.load_weights(UNET_WEIGHTS)
    model = load_model(UNET_WEIGHTS, compile=False)  # ← 整模型加载
    # read png and ready for predict
    images = []
    for files in os.listdir(imagepath):
        tempdir = os.path.join(imagepath, files)
        img = cv2.imread(tempdir, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    for index, img in enumerate(images):
        img = prepare_image_for_net(img)
        images[index] = img
    images3d = np.vstack(images)
    y_pred = model.predict(images3d, batch_size=BATCH_SIZE)
    count = 0
    for y in y_pred:
        y *= 255.
        y = y.reshape((y.shape[0], y.shape[1])).astype(np.uint8)
        cv2.imwrite(os.path.join(maskpath, os.listdir(imagepath)[count]), y)   # 将分割结果保存
        count += 1
    print(count)


def plot_one_box(img, coord, label=None, line_thickness=None):
    """
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.矩形线条粗细
    """
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = [0, 0, 255]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))  # 中心点，宽高
    # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩形
    # img是原图（x，y）是矩阵的左上点坐标（x+w，y+h）是矩阵的右下点坐标
    # （0,255,0）是画线对应的rgb颜色2是所画的线的宽度
    img = cv2.rectangle(img, c1, c2, color, thickness=tl)
    # 在矩形框上显示出类别
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img


if __name__ == "__main__":

    test_img_path = r'./Unet_dataset_test/images'
    # 2) 保存 mask 的目录
    mask_path = r"./demo_results/masks"       # <== 事先 mkdir -p
    # 3) （可选）把红框画回原图
    detect_path = r"./demo_results/boxed"     # <== 若不用红框就注释掉下面整段
    unet_predict(test_img_path, mask_path)

    """ 将原图和分割得到的结节坐标结合，框出结节位置 """
    for files in os.listdir(mask_path):
        centers = unet_candidate_dicom(os.path.join(mask_path, files))  # 获得mask的结节中心坐标
        # print('y, x', centers)
        img_ori = cv2.imread(os.path.join(test_img_path, files))        # 读取原始图片
        for i in range(len(centers)):
            box = [centers[i][1]-5.5, centers[i][0]-5.5, centers[i][1]+5.5, centers[i][0]+5.5]
            img_ori = plot_one_box(img_ori, box)     # 得到标注后的图片
        cv2.imwrite(os.path.join(detect_path, files), img_ori)

