"""
LUNA16 数据集预处理文件
处理.mhd格式文件，生成320x320的训练图像和mask
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from tqdm import tqdm
import csv
import glob
from pathlib import Path  # ⬅️ 放到文件顶部的 import 区

class LUNA16Preprocessor:
    """LUNA16数据集预处理类"""

    def __init__(self, data_dir, annotations_file, output_dir, subset_list=None):
        """
        初始化预处理器

        Args:
            data_dir: LUNA16数据集根目录
            annotations_file: annotations.csv文件路径
            output_dir: 输出目录
            subset_list: 要处理的subset列表，如[0,1,2]，None表示处理所有
        """
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.subset_list = subset_list

        # 固定参数
        self.img_size = 320  # 输出图像大小
        self.mask_margin = 5  # mask边缘扩展像素

        # 创建输出目录
        self._create_output_dirs()

        # 加载标注数据
        self.annotations = pd.read_csv(annotations_file)
        print(f"Loaded {len(self.annotations)} annotations")

    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = ['images', 'masks']
        for dir_name in dirs:
            dir_path = os.path.join(self.output_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

    def load_mhd_file(self, file_path):
        """
        加载.mhd文件

        Args:
            file_path: .mhd文件路径

        Returns:
            img_array: 3D图像数组
            origin: 原点坐标
            spacing: 体素间距
        """
        itk_img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        return img_array, origin, spacing

    def normalize_hu(self, image, min_bound=-1000.0, max_bound=400.0):
        """
        将HU值归一化到[0,1]区间

        Args:
            image: 输入图像
            min_bound: HU值下界（肺窗）
            max_bound: HU值上界（肺窗）

        Returns:
            归一化后的图像
        """
        image = (image - min_bound) / (max_bound - min_bound)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    def world_to_voxel(self, world_coord, origin, spacing):
        """
        将世界坐标转换为体素坐标

        Args:
            world_coord: 世界坐标 [x, y, z]
            origin: 原点
            spacing: 间距

        Returns:
            体素坐标 [z, y, x] (注意顺序)
        """
        # SimpleITK的坐标顺序是 [x, y, z]
        # numpy数组的顺序是 [z, y, x]
        voxel_coord = (world_coord - origin) / spacing
        # 返回 [z, y, x] 顺序
        return np.array([voxel_coord[2], voxel_coord[1], voxel_coord[0]])

    def make_mask(self, center, diameter, width, height):
        """
        生成圆形结节mask

        Args:
            center: 结节中心坐标 (x, y)
            diameter: 结节直径（像素）
            width: 图像宽度
            height: 图像高度

        Returns:
            mask数组
        """
        mask = np.zeros([height, width], dtype=np.uint8)

        # 计算mask边界
        radius = diameter / 2 + self.mask_margin
        y, x = np.ogrid[:height, :width]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_circle] = 1

        return mask

    def resample_image(self, image, old_spacing, new_spacing=[1, 1, 1]):
        """
        重采样图像到新的spacing

        Args:
            image: 3D图像
            old_spacing: 原始spacing
            new_spacing: 目标spacing

        Returns:
            重采样后的图像
        """
        resize_factor = old_spacing / new_spacing
        new_shape = np.round(image.shape * resize_factor)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor

        image = ndi.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def process_series(self, series_uid, mhd_file):
        """
        处理单个序列

        Args:
            series_uid: 序列ID
            mhd_file: .mhd文件路径
        """
        print(f"\nProcessing series: {series_uid}")

        # 加载.mhd文件
        img_array, origin, spacing = self.load_mhd_file(mhd_file)
        print(f"  Original shape: {img_array.shape}, spacing: {spacing}")

        # 重采样到1mm spacing（可选）
        # img_array, spacing = self.resample_image(img_array, spacing)

        # 获取该序列的所有标注
        series_annotations = self.annotations[
            self.annotations['seriesuid'] == series_uid
        ]

        # 处理每个切片
        for slice_idx in tqdm(range(img_array.shape[0]), desc="Processing slices"):
            # 获取切片
            slice_img = img_array[slice_idx]

            # HU值归一化
            slice_img_norm = self.normalize_hu(slice_img)

            # 调整大小到320x320
            slice_img_resized = cv2.resize(
                slice_img_norm,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_CUBIC
            )

            # 初始化空mask
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

            # 如果有标注，生成结节mask
            if not series_annotations.empty:
                for _, nodule in series_annotations.iterrows():
                    # 世界坐标
                    world_coord = np.array([
                        nodule['coordX'],
                        nodule['coordY'],
                        nodule['coordZ']
                    ])

                    # 转换为体素坐标
                    voxel_coord = self.world_to_voxel(world_coord, origin, spacing)

                    # 检查结节是否在当前切片（允许一定范围的误差）
                    if abs(voxel_coord[0] - slice_idx) <= 2:  # 增加容差
                        # 计算在调整后图像中的位置
                        # voxel_coord是[z, y, x]格式
                        scale_y = self.img_size / slice_img.shape[0]
                        scale_x = self.img_size / slice_img.shape[1]

                        center_x = int(voxel_coord[2] * scale_x)
                        center_y = int(voxel_coord[1] * scale_y)

                        # 计算直径（mm转像素）
                        # spacing是[x, y, z]格式
                        diameter_pixels = int(nodule['diameter_mm'] / spacing[0] * scale_x)

                        # 确保坐标在有效范围内
                        if 0 <= center_x < self.img_size and 0 <= center_y < self.img_size:
                            # 生成结节mask
                            nodule_mask = self.make_mask(
                                (center_x, center_y),
                                diameter_pixels,
                                self.img_size,
                                self.img_size
                            )
                            # 合并到总mask
                            mask = np.maximum(mask, nodule_mask)

                            # 调试信息
                            if nodule_mask.max() > 0:
                                print(f"    Found nodule at slice {slice_idx}: center=({center_x}, {center_y}), diameter={diameter_pixels}px")

            # 保存图像和mask
            img_filename = f"{series_uid}_slice_{slice_idx:04d}_img.png"
            mask_filename = f"{series_uid}_slice_{slice_idx:04d}_mask.png"

            cv2.imwrite(
                os.path.join(self.output_dir, 'images', img_filename),
                (slice_img_resized * 255).astype(np.uint8)
            )
            cv2.imwrite(
                os.path.join(self.output_dir, 'masks', mask_filename),
                (mask * 255).astype(np.uint8)
            )

    def process_all(self):
        """处理所有数据"""
        # 确定要处理的subset
        if self.subset_list is None:
            subset_list = list(range(10))  # LUNA16有10个subset (0-9)
        else:
            subset_list = self.subset_list

        total_nodules_found = 0

        # 处理每个subset
        for subset_idx in subset_list:
            subset_dir = os.path.join(self.data_dir, f'subset{subset_idx}')

            if not os.path.exists(subset_dir):
                print(f"Subset directory not found: {subset_dir}")
                continue

            print(f"\n{'='*50}")
            print(f"Processing subset{subset_idx}")
            print(f"{'='*50}")

            # 获取所有.mhd文件
            mhd_files = glob.glob(os.path.join(subset_dir, "*.mhd"))
            print(f"Found {len(mhd_files)} .mhd files")

            # 处理每个.mhd文件
            for mhd_file in mhd_files:
                # 从文件名提取series UID
                series_uid = os.path.basename(mhd_file).replace('.mhd', '')

                try:
                    self.process_series(series_uid, mhd_file)
                except Exception as e:
                    print(f"Error processing {series_uid}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        print("\n预处理完成！")
        self._generate_file_lists()

    def _generate_file_lists(self):
        """生成训练集和验证集文件列表"""
        # 获取所有图像文件
        images_dir = os.path.join(self.output_dir, 'images')
        all_images = glob.glob(os.path.join(images_dir, '*.png'))
        all_images.sort()

        print(f"\n在 {images_dir} 找到 {len(all_images)} 个图像文件")

        # 创建图像-mask对
        file_pairs = []
        for img_path in all_images:
            # 使用os.path来处理路径，确保跨平台兼容
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace('_img.png', '_mask.png')
            mask_path = os.path.join(self.output_dir, 'masks', mask_name)

            if os.path.exists(mask_path):
                file_pairs.append((img_path, mask_path))

        # 分离正负样本
        positive_pairs = []
        negative_pairs = []

        print("\n正在统计正负样本...")
        for img_path, mask_path in tqdm(file_pairs):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                positive_pairs.append((img_path, mask_path))
            else:
                negative_pairs.append((img_path, mask_path))

        print(f"\n统计信息：")
        print(f"总切片数: {len(file_pairs)}")
        print(f"阳性切片数: {len(positive_pairs)}")
        print(f"阴性切片数: {len(negative_pairs)}")

        # 8:2划分训练集和验证集
        train_ratio = 0.8

        # 分别对正负样本进行划分，确保比例均衡
        n_pos_train = int(len(positive_pairs) * train_ratio)
        n_neg_train = int(len(negative_pairs) * train_ratio)

        train_pairs = positive_pairs[:n_pos_train] + negative_pairs[:n_neg_train]
        val_pairs = positive_pairs[n_pos_train:] + negative_pairs[n_neg_train:]

        # 打乱顺序
        np.random.shuffle(train_pairs)
        np.random.shuffle(val_pairs)

        # 保存文件列表
        self._save_file_list(train_pairs, 'train_uid_list.csv')
        self._save_file_list(val_pairs, 'val_uid_list.csv')

        print(f"\n文件列表已生成：")
        print(f"训练集: {len(train_pairs)} 对")
        print(f"验证集: {len(val_pairs)} 对")


    def _save_file_list(self, file_pairs, filename):
        """保存文件列表到CSV"""
        csv_path = os.path.join(self.output_dir, filename)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for img_path, mask_path in file_pairs:
                writer.writerow([Path(img_path).as_posix(),
                                 Path(mask_path).as_posix()])


    def visualize_samples(self, num_samples=5):
        """可视化一些样本，验证mask是否正确"""
        import matplotlib.pyplot as plt

        # 读取文件列表
        train_csv = os.path.join(self.output_dir, 'train_uid_list.csv')
        if not os.path.exists(train_csv):
            print("请先运行process_all()生成数据")
            return

        # 读取一些正样本
        positive_samples = []
        with open(train_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    img_path, mask_path = row[0], row[1]
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None and mask.max() > 0:
                        positive_samples.append((img_path, mask_path))
                        if len(positive_samples) >= num_samples:
                            break

        if not positive_samples:
            print("没有找到正样本")
            return

        # 可视化
        fig, axes = plt.subplots(len(positive_samples), 3, figsize=(12, 4*len(positive_samples)))
        if len(positive_samples) == 1:
            axes = axes.reshape(1, -1)

        for idx, (img_path, mask_path) in enumerate(positive_samples):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 创建叠加图
            overlay = img.copy()
            overlay[mask > 0] = 255

            axes[idx, 0].imshow(img, cmap='gray')
            axes[idx, 0].set_title(f'Image: {os.path.basename(img_path)}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title('Mask')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(overlay, cmap='gray')
            axes[idx, 2].set_title('Overlay')
            axes[idx, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'sample_visualizations.png')
        plt.savefig(save_path)
        print(f"可视化结果已保存到: {save_path}")
        plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='LUNA16 数据预处理')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='LUNA16数据集根目录')
    parser.add_argument('--annotations', type=str, required=True,
                       help='annotations.csv文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--subsets', type=int, nargs='+', default=None,
                       help='要处理的subset列表，如: 0 1 2')

    args = parser.parse_args()

    # 创建预处理器
    preprocessor = LUNA16Preprocessor(
        data_dir=args.data_dir,
        annotations_file=args.annotations,
        output_dir=args.output_dir,
        subset_list=args.subsets
    )

    # 执行预处理
    preprocessor.process_all()


if __name__ == '__main__':
    # 命令行使用示例：
    # python luna16_preprocess.py --data_dir /path/to/LUNA16 --annotations /path/to/annotations.csv --output_dir ./preprocessed --subsets 0

    # 直接运行示例（请根据实际路径修改）：
    if len(sys.argv) == 1:  # 没有命令行参数时的示例
        import sys
        print("示例用法：")
        print("python luna16_preprocess.py --data_dir /path/to/LUNA16 --annotations /path/to/annotations.csv --output_dir ./preprocessed")
        print("\n或者修改代码中的路径直接运行：")

        # 如果要直接运行，请取消下面的注释并修改路径
        preprocessor = LUNA16Preprocessor(
            data_dir='D:/LUNA16',  # 修改为您的LUNA16路径
            annotations_file='D:/LUNA16/annotations.csv',  # 修改为您的标注文件路径
            output_dir='./preprocessed',
            subset_list=[0,1,2]  # 先处理subset0测试
        )
        preprocessor.process_all()
    else:
        main()