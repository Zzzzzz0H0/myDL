import numpy as np


def compute_iou(pred, gt, result):
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    # GPU->CPU tensor->array
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    # 对每一类来说
    for i in range(8):
        #计算实际像素点和预测像素点的个数
        single_gt = gt==i
        single_pred = pred==i
        # 计算交集,值都为0或1，只有都为1的矩阵乘法才能为1
        temp_tp = np.sum(single_gt * single_pred)
        # 计算并集
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    # 返回每一类的交并集
    return result
