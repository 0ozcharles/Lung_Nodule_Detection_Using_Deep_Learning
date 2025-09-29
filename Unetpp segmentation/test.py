import matplotlib.pyplot as plt
import numpy as np

# 导入你的train_unet_1中定义的函数和变量
from train_unet_1 import get_balanced_batch_generator, IMG_SIZE

def debug_val_masks():
    print("🔍 开始验证集样本调试...")
    val_gen, _ = get_balanced_batch_generator("val_uid_list.csv", batch_size=8, train_mode=False)
    imgs, masks = next(val_gen)

    for i in range(8):
        print(f"[样本{i}] mask 最大值: {masks[i].max()}, 像素和: {masks[i].sum()}")
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(imgs[i, :, :, 0], cmap='gray')
        plt.title("Image")
        plt.subplot(1, 2, 2)
        plt.imshow(masks[i, :, :, 0], cmap='gray')
        plt.title("Mask")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    debug_val_masks()
