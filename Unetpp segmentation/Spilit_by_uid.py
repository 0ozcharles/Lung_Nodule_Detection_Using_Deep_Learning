#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版数据集划分脚本
主要修正：
1. 确保验证集有足够的正样本
2. 保持结节大小分布的平衡
3. 避免数据泄露
4. 优化训练集平衡策略
"""
import csv
import glob
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# 配置参数
IMG_DIR = Path('Unet_dataset/images')
MASK_DIR = Path('Unet_dataset/masks')
TRAIN_CSV = 'train_uid_list.csv'
VAL_CSV = 'val_uid_list.csv'

VAL_RATIO = 0.15  # 减小验证集比例，增加训练数据
MIN_POS_SAMPLES_IN_VAL = 50  # 确保验证集有足够正样本
TRAIN_NEG_POS_RATIO = 2.5  # 适度的负正比例
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def analyze_case_statistics():
    """分析每个病例的统计信息"""
    case_stats = defaultdict(lambda: {
        'total_slices': 0,
        'positive_slices': 0,
        'total_nodule_pixels': 0,
        'nodule_sizes': [],
        'slice_indices': []
    })

    print("分析病例统计信息...")

    for img_path in IMG_DIR.glob('*_img.png'):
        # 提取病例ID - 更严格的解析
        filename = img_path.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            case_id = parts[0]  # 只取第一部分作为case_id
            slice_idx = int(parts[1]) if parts[1].isdigit() else 0
        else:
            continue

        mask_path = MASK_DIR / img_path.name.replace('_img', '_mask')

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), 0)
            case_stats[case_id]['total_slices'] += 1
            case_stats[case_id]['slice_indices'].append(slice_idx)

            if mask.max() > 0:
                case_stats[case_id]['positive_slices'] += 1
                nodule_pixels = np.sum(mask > 0)
                case_stats[case_id]['total_nodule_pixels'] += nodule_pixels
                case_stats[case_id]['nodule_sizes'].append(nodule_pixels)

    # 清理统计信息并分类
    positive_cases = []
    negative_cases = []

    for case_id, stats in case_stats.items():
        if stats['positive_slices'] > 0:
            # 计算结节特征
            avg_nodule_size = np.mean(stats['nodule_sizes']) if stats['nodule_sizes'] else 0
            max_nodule_size = max(stats['nodule_sizes']) if stats['nodule_sizes'] else 0
            nodule_density = stats['positive_slices'] / stats['total_slices']

            positive_cases.append({
                'case_id': case_id,
                'positive_slices': stats['positive_slices'],
                'total_slices': stats['total_slices'],
                'avg_nodule_size': avg_nodule_size,
                'max_nodule_size': max_nodule_size,
                'nodule_density': nodule_density,
                'total_nodule_pixels': stats['total_nodule_pixels']
            })
        else:
            negative_cases.append({
                'case_id': case_id,
                'total_slices': stats['total_slices']
            })

    print(f"\n病例统计:")
    print(f"  正样本病例: {len(positive_cases)}")
    print(f"  负样本病例: {len(negative_cases)}")
    print(f"  总病例数: {len(case_stats)}")

    return positive_cases, negative_cases, case_stats


def stratified_split_improved(positive_cases, negative_cases):
    """改进的分层划分策略"""

    # 按结节大小和密度进行多维度分层
    positive_cases.sort(key=lambda x: (x['avg_nodule_size'], x['nodule_density']))

    val_positive_cases = []
    train_positive_cases = []

    # 确保验证集包含各种类型的结节
    n_pos = len(positive_cases)

    # 三个层次：小结节、中结节、大结节
    small_nodules = [c for c in positive_cases if c['avg_nodule_size'] < 50]
    medium_nodules = [c for c in positive_cases if 50 <= c['avg_nodule_size'] < 200]
    large_nodules = [c for c in positive_cases if c['avg_nodule_size'] >= 200]

    print(f"\n结节大小分布:")
    print(f"  小结节病例 (<50像素): {len(small_nodules)}")
    print(f"  中结节病例 (50-200像素): {len(medium_nodules)}")
    print(f"  大结节病例 (>=200像素): {len(large_nodules)}")

    # 从每个组中按比例选择验证集
    for group, name in [(small_nodules, '小'), (medium_nodules, '中'), (large_nodules, '大')]:
        if len(group) > 0:
            random.shuffle(group)
            n_val = max(1, int(len(group) * VAL_RATIO))
            val_positive_cases.extend(group[:n_val])
            train_positive_cases.extend(group[n_val:])
            print(f"  {name}结节 - 训练: {len(group[n_val:])}, 验证: {n_val}")

    # 确保验证集有足够的正样本
    total_val_pos_slices = sum(c['positive_slices'] for c in val_positive_cases)
    if total_val_pos_slices < MIN_POS_SAMPLES_IN_VAL:
        needed_slices = MIN_POS_SAMPLES_IN_VAL - total_val_pos_slices

        # 从训练集中选择更多病例
        train_positive_cases.sort(key=lambda x: x['positive_slices'], reverse=True)
        additional_cases = []

        for case in train_positive_cases:
            if needed_slices <= 0:
                break
            additional_cases.append(case)
            needed_slices -= case['positive_slices']

        val_positive_cases.extend(additional_cases)
        train_positive_cases = [c for c in train_positive_cases if c not in additional_cases]

    # 负样本病例划分
    random.shuffle(negative_cases)
    n_val_neg = max(1, int(len(negative_cases) * VAL_RATIO))
    val_negative_cases = negative_cases[:n_val_neg]
    train_negative_cases = negative_cases[n_val_neg:]

    print(f"\n最终病例级别划分:")
    print(f"  训练集: {len(train_positive_cases)} 正病例, {len(train_negative_cases)} 负病例")
    print(f"  验证集: {len(val_positive_cases)} 正病例, {len(val_negative_cases)} 负病例")

    # 统计验证集正样本数量
    val_pos_slices = sum(c['positive_slices'] for c in val_positive_cases)
    print(f"  验证集正样本切片: {val_pos_slices}")

    return (train_positive_cases, train_negative_cases,
            val_positive_cases, val_negative_cases)


def collect_slices_by_cases(cases):
    """根据病例收集切片"""
    all_slices = []

    for case_info in cases:
        case_id = case_info['case_id']

        # 查找该病例的所有切片
        pattern = f'{case_id}_*_img.png'
        case_files = list(IMG_DIR.glob(pattern))

        for img_path in case_files:
            mask_path = MASK_DIR / img_path.name.replace('_img', '_mask')
            if mask_path.exists():
                all_slices.append([str(img_path), str(mask_path)])

    return all_slices


def balance_training_set_improved(train_slices):
    """改进的训练集平衡策略"""
    positive_slices = []
    negative_slices = []

    print("\n分析训练集切片...")
    for img_path, mask_path in train_slices:
        mask = cv2.imread(mask_path, 0)
        if mask.max() > 0:
            positive_slices.append([img_path, mask_path])
        else:
            negative_slices.append([img_path, mask_path])

    print(f"  原始分布: {len(positive_slices)} 正, {len(negative_slices)} 负")

    # 智能负样本选择
    if len(negative_slices) > len(positive_slices) * TRAIN_NEG_POS_RATIO:
        target_neg = int(len(positive_slices) * TRAIN_NEG_POS_RATIO)

        # 随机选择负样本，确保多样性
        random.shuffle(negative_slices)
        negative_slices = negative_slices[:target_neg]

        print(f"  平衡后: {len(positive_slices)} 正, {len(negative_slices)} 负")
        print(f"  负:正比例: {len(negative_slices) / len(positive_slices):.1f}:1")

    return positive_slices + negative_slices


def validate_split(train_cases, val_cases):
    """验证划分结果，确保无数据泄露"""
    train_case_ids = set()
    val_case_ids = set()

    for case_list, case_set in [(train_cases, train_case_ids), (val_cases, val_case_ids)]:
        for case_info in case_list:
            case_set.add(case_info['case_id'])

    overlap = train_case_ids.intersection(val_case_ids)
    if overlap:
        print(f"❌ 错误：发现{len(overlap)}个重叠病例：{overlap}")
        return False
    else:
        print(f"✅ 验证通过：训练集和验证集无病例重叠")
        return True


def analyze_final_distribution(csv_file, name):
    """分析最终数据分布"""
    total, pos, neg = 0, 0, 0
    nodule_sizes = []

    with open(csv_file, 'r') as f:
        for img_path, mask_path in csv.reader(f):
            mask = cv2.imread(mask_path, 0)
            total += 1
            if mask.max() > 0:
                pos += 1
                nodule_sizes.append(np.sum(mask > 0))
            else:
                neg += 1

    print(f"\n{name}最终统计:")
    print(f"  总切片: {total}")
    print(f"  正样本: {pos} ({pos / total * 100:.1f}%)")
    print(f"  负样本: {neg} ({neg / total * 100:.1f}%)")

    if nodule_sizes:
        print(f"  结节大小分布:")
        print(f"    最小: {min(nodule_sizes)} 像素")
        print(f"    最大: {max(nodule_sizes)} 像素")
        print(f"    平均: {np.mean(nodule_sizes):.1f} 像素")
        print(f"    中位数: {np.median(nodule_sizes):.1f} 像素")

        # 分析大小分布
        small = sum(1 for s in nodule_sizes if s < 50)
        medium = sum(1 for s in nodule_sizes if 50 <= s < 200)
        large = sum(1 for s in nodule_sizes if s >= 200)

        print(f"    小结节(<50): {small} ({small / len(nodule_sizes) * 100:.1f}%)")
        print(f"    中结节(50-200): {medium} ({medium / len(nodule_sizes) * 100:.1f}%)")
        print(f"    大结节(>=200): {large} ({large / len(nodule_sizes) * 100:.1f}%)")

    return {'total': total, 'positive': pos, 'negative': neg, 'nodule_sizes': nodule_sizes}


def main():
    """主函数"""
    print("=" * 60)
    print("改进的数据集划分脚本")
    print("=" * 60)

    # 1. 分析病例统计
    positive_cases, negative_cases, case_stats = analyze_case_statistics()

    if len(positive_cases) == 0:
        print("❌ 错误：没有找到正样本病例，请检查数据预处理")
        return

    # 2. 改进的分层划分
    train_pos_cases, train_neg_cases, val_pos_cases, val_neg_cases = \
        stratified_split_improved(positive_cases, negative_cases)

    # 3. 验证划分结果
    all_train_cases = train_pos_cases + train_neg_cases
    all_val_cases = val_pos_cases + val_neg_cases

    if not validate_split(all_train_cases, all_val_cases):
        print("❌ 数据划分验证失败，退出")
        return

    # 4. 收集切片
    print("\n收集切片...")
    train_slices = collect_slices_by_cases(all_train_cases)
    val_slices = collect_slices_by_cases(all_val_cases)

    print(f"  训练集原始切片: {len(train_slices)}")
    print(f"  验证集切片: {len(val_slices)}")

    # 5. 平衡训练集
    train_slices = balance_training_set_improved(train_slices)

    # 打乱数据
    random.shuffle(train_slices)
    random.shuffle(val_slices)

    # 6. 保存CSV
    print(f"\n保存数据集...")
    with open(TRAIN_CSV, 'w', newline='') as f:
        csv.writer(f).writerows(train_slices)

    with open(VAL_CSV, 'w', newline='') as f:
        csv.writer(f).writerows(val_slices)

    print(f"✅ 数据集已保存:")
    print(f"   {TRAIN_CSV}: {len(train_slices)} 样本")
    print(f"   {VAL_CSV}: {len(val_slices)} 样本")

    # 7. 分析最终分布
    train_stats = analyze_final_distribution(TRAIN_CSV, "训练集")
    val_stats = analyze_final_distribution(VAL_CSV, "验证集")

    # 8. 保存划分信息
    split_info = {
        'train_cases': [c['case_id'] for c in all_train_cases],
        'val_cases': [c['case_id'] for c in all_val_cases],
        'statistics': {
            'train': train_stats,
            'validation': val_stats
        },
        'parameters': {
            'val_ratio': VAL_RATIO,
            'train_neg_pos_ratio': TRAIN_NEG_POS_RATIO,
            'min_pos_samples_in_val': MIN_POS_SAMPLES_IN_VAL
        }
    }

    with open('dataset_split_info_fixed.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # 9. 质量检查和建议
    print(f"\n质量检查:")

    if val_stats['positive'] < 30:
        print(f"⚠️ 警告：验证集正样本过少 ({val_stats['positive']})，可能影响评估准确性")

    train_ratio = train_stats['negative'] / max(1, train_stats['positive'])
    if train_ratio > 5:
        print(f"⚠️ 警告：训练集负样本比例过高 ({train_ratio:.1f}:1)")
    elif train_ratio < 1:
        print(f"⚠️ 警告：训练集正样本比例过高 ({1 / train_ratio:.1f}:1)")
    else:
        print(f"✅ 训练集正负样本比例合理 ({train_ratio:.1f}:1)")

    print(f"\n建议:")
    print(f"  1. 检查验证集样本质量，确保有代表性")
    print(f"  2. 训练时监控验证集指标，防止过拟合")
    print(f"  3. 可视化几个验证集样本，确保标注正确")

    print("\n" + "=" * 60)
    print("数据集划分完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()