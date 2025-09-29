import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import roc_curve, auc


"""
len of tst_fold_data_0 is: 6574
len of tst_fold_data_1 is: 7581   # 1例阳性
len of tst_fold_data_2 is: 8455   # 8例阳性  
len of tst_fold_data_3 is: 7451   # 15例阳性
len of tst_fold_data_4 is: 8362   # 29例阳性
test result's shape: (38423,)
"""


def plot_roc_curve(label, predict,
                   save_path="roc_curve.png",
                   show=False,
                   dpi=100):
    """
    绘制并保存 ROC 曲线

    Parameters
    ----------
    label : array-like
        真实标签（0/1）
    predict : array-like
        预测得分（越大越偏正类）
    save_path : str
        保存文件路径（支持 .png/.pdf/.svg 等 matplotlib 支持的格式）
    show : bool
        是否同时在屏幕上显示
    dpi : int
        输出分辨率
    """
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()  # 释放内存



if __name__ == '__main__':
    # pred = np.load("./test_sum/pred_result/0_0_pm.npy")
    # _, _, tst_dat1, tst_lbl1 = utils.load_fold(fold_num=0)
    # _, _, tst_dat2, tst_lbl2 = utils.load_fold(fold_num=1)
    # _, _, tst_dat3, tst_lbl3 = utils.load_fold(fold_num=2)
    # _, _, tst_dat4, tst_lbl4 = utils.load_fold(fold_num=3)
    # _, _, tst_dat5, tst_lbl5 = utils.load_fold(fold_num=4)
    #
    # all_tst_lbl = [tst_lbl1, tst_lbl2, tst_lbl3, tst_lbl4, tst_lbl5]
    # all_tst_lbl = np.concatenate(all_tst_lbl, axis=0)

    """
    # 统计每个fold的TP个数
    index_1 = np.argwhere(all_tst_lbl[6574:7581] == 1)
    print(index_1[:, 0], "  ", len(index_1[:, 0]), "\n")
    index_2 = np.argwhere(all_tst_lbl[7581:8455+7581] == 1)
    print(index_2[:, 0], "  ", len(index_2[:, 0]), "\n")
    index_3 = np.argwhere(all_tst_lbl[8455+7581:7451+8455+7581] == 1)
    print(index_3[:, 0], "  ", len(index_3[:, 0]), "\n")
    index_4 = np.argwhere(all_tst_lbl[7451+8455+7581:8362+7451+8455+7581] == 1)
    print(index_4[:, 0], "  ", len(index_4[:, 0]), "\n")
    """

    # plot_roc_curve(all_tst_lbl[8455+7581:7451+8455+7581], pred[8455+7581:7451+8455+7581])
    # plot_roc_curve(all_tst_lbl[7581:8455+7581], pred[7581:8455+7581])
    # plot_roc_curve(all_tst_lbl[6574:7581], pred[6574:7581])
    # plot_roc_curve(all_tst_lbl[0:6574], pred[0:6574])
    # plot_roc_curve(all_tst_lbl, pred)

   # 只加载一次整个 subset3 测试集（在 --test 模式下 load_fold 已经返回了它）

    scores = np.load("./test_sum/pred_result/0_0_pm.npy")
    labels = np.load("./test_sum/pred_result/0_0_lbl.npy")
    plot_roc_curve(labels, scores,
                   save_path="./test_sum/roc_curve_subset3.png",
                   show=False)

