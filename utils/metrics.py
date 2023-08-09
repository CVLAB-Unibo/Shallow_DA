import numpy as np

class mIoU():
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, c_num):
        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._num_class = c_num

    def reset(self):
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf):
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs)
    
    def get_score(self, mask=None):
        ious = np.nan_to_num(self._per_cls_iou)
        if mask is not None:
            ious = ious[mask]
        return np.mean(ious) * 100, ious*100