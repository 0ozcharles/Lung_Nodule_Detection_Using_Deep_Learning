import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import roc_curve, auc
import os


def plot_roc_curve(label, predict, save_path=None):
    """绘制并（可选）保存 ROC 曲线"""
    fpr, tpr, _ = roc_curve(label, predict)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="red", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(f"AUC: {roc_auc:.3f}", fontsize=16)
    plt.legend(loc="lower right", fontsize=16)

    # === 关键：保存文件 ===
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    pred = np.load(
        r"F:\Pulmonary-Nodule-Detection-Based-on-Deep-Learning\Reduce false positive\test_sum\pred_result\0_0_pm.npy"
    )

    _, _, _, tst_lbl1 = utils.load_fold(fold_num=0)
    _, _, _, tst_lbl2 = utils.load_fold(fold_num=1)
    _, _, _, tst_lbl3 = utils.load_fold(fold_num=2)
    _, _, _, tst_lbl4 = utils.load_fold(fold_num=3)
    _, _, _, tst_lbl5 = utils.load_fold(fold_num=4)

    all_tst_lbl = np.concatenate([tst_lbl1, tst_lbl2, tst_lbl3, tst_lbl4, tst_lbl5], axis=0)

    # 例：保存 Fold-0 的 ROC
    plot_roc_curve(
        all_tst_lbl[:6574],  # fold-0 真值
        pred[:6574],  # fold-0 预测
        save_path=r"F:\ROC_figs\roc_fold0.png",  # 想存到哪就写哪
    )

    # 例：保存整体 ROC
    plot_roc_curve(
        all_tst_lbl,
        pred,
        save_path=r"F:\ROC_figs\roc_all.png",
    )
