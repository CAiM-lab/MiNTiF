# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import tensorflow as tf
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import skimage
import skimage.feature
from scipy.ndimage import measurements

eps = 1e-10


def calculate_precision(tp, fp):
    return tp / (tp + fp + eps)


def calculate_recall(tp, fn):
    return tp / (tp + fn + eps)


def calculate_fscore(tp, fn, fp):
    return 2.0 * tp / (2.0 * tp + fn + fp + eps)


def calculate_iou(tp, fn, fp):
    return tp / (tp + fn + fp + eps)


class CellMetrics:  # (tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(CellMetrics, self).__init__(**kwargs)
        self.mindist = 5
        self.detect_threshold = 15
        self.crop_border = 4
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.fscore = 0
        self.iou = 0
        self.derror_pos = []
        self.derror_all = []
        self.derror_pos_mean = 0
        self.derror_pos_std = 0
        self.derror_pos_95ptl = 0
        self.derror_mean = 0
        self.derror_std = 0
        self.derror_95ptl = 0

    def __call__(self, y_true, y_pred):
        self.update_state(y_true, y_pred)

    def peak_detect(self, vol):
        """
        Detects coordinates in a density map.

        It is important to find the right parameters for peak_local_max. I normally set min_distance and exclude_border
        to the cell radius (5um in this case). The threshold may imply a bit of tuning, but I currently use 15.
        :param vol: density map
        :return: coordinates
        """

        mask = skimage.feature.peak_local_max(vol,
                                              min_distance=self.mindist,
                                              threshold_abs=self.detect_threshold,
                                              exclude_border=self.crop_border,
                                              indices=False)
        labels = measurements.label(mask)[0]
        merged_peaks = measurements.center_of_mass(mask, labels, range(1, np.max(labels) + 1))
        merged_peaks = np.array(merged_peaks)
        return merged_peaks

    def gt_match(self, Xgt, Xpred):
        """
        Function to calculate the quality of spot detection
        :param Xgt: the ground truth coordinates
        :param Xpred: the predicted coordinates
        :param rmatch: the threshold distance below which we count a positive
        :return:
        """
        rmatch = self.mindist
        if len(Xgt) == 0:
            qmetrics = {'TP': np.nan,
                        'FN': 0,
                        'FP': len(Xpred),
                        'precision': np.nan,
                        'recall': np.nan,
                        'fscore': np.nan,
                        'derror_pos': None,
                        'derror_all': None}
        elif len(Xpred) == 0:
            qmetrics = {'TP': 0,
                        'FN': len(Xgt),
                        'FP': 0,
                        'precision': 0,
                        'recall': 0,
                        'fscore': 0,
                        # 'derror_pos': np.nan,
                        # 'derror_all': np.nan
                        'derror_pos': None,
                        'derror_all': None
                        }
        else:
            # Hungarian algorithm
            C = distance_matrix(Xgt, Xpred)
            # to avoid 0 entries
            C[C==0]+=10e-6
            Ctop = np.log(C)
            igt, ipred = linear_sum_assignment(Ctop)
            nngt = np.empty((len(Xgt),), dtype=np.intp)
            nngt.fill(-1)
            nnpred = np.empty((len(Xpred),), dtype=np.intp)
            nnpred.fill(-1)
            for pgt, ppred in zip(igt, ipred):
                indval = C[pgt, ppred]
                if indval <= rmatch:
                    nngt[pgt] = indval
                    nnpred[ppred] = indval

            tp = sum(nnpred > -1)
            fn = sum(nngt == -1)
            fp = sum(nnpred == -1)
            derror_all = C.min(axis=0)
            qmetrics = {'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'precision': tp / (tp + fp),
                        'recall': tp / (tp + fn),
                        'fscore': 2 * tp / (2 * tp + fp + fn),
                        'derror_pos': C[igt, ipred],
                        'derror_all': derror_all}
        return qmetrics

    # @tf.function
    def update_state(self, y_true, y_pred):
        ynp_true = np.array(y_true)

        # predicted DMs
        ynp_pred = np.array(y_pred)

        # loop trough different masks
        for bb in range(ynp_pred.shape[0]):
            # discard last dim
            vol = ynp_pred[bb, ..., 0]
            # get all non negative coordinates
            coor_gt = ynp_true[bb, ..., 0]
            coor_gt = coor_gt[(coor_gt > 0).any(axis=1)]
            # discard coordinates on border
            indscrop = ~((coor_gt < self.crop_border).any(axis=1) | (
                    coor_gt > (np.array(vol.shape) - self.crop_border)).any(axis=1))
            coor_gt = coor_gt[indscrop]
            coor_pred = self.peak_detect(vol)
            dmetrics = self.gt_match(coor_gt, coor_pred)
            if not np.isnan(dmetrics['TP']):
                self.tp += dmetrics['TP']
            self.fp += dmetrics['FP']
            self.fn += dmetrics['FN']
            if dmetrics['derror_pos'] is not None:  # and   not np.isnan(dmetrics['derror_pos']):
                self.derror_pos = np.append(self.derror_pos, dmetrics['derror_pos'])
            if dmetrics['derror_all'] is not None:  # and not np.isnan(dmetrics['derror_all']):
                self.derror_all = np.append(self.derror_all, dmetrics['derror_all'])
        self.precision = calculate_precision(self.tp, self.fp)
        self.recall = calculate_recall(self.tp, self.fn)
        self.fscore = calculate_fscore(self.tp, self.fn, self.fp)
        self.iou = calculate_iou(self.tp, self.fn, self.fp)

        self.derror_pos_mean = tf.reduce_mean(self.derror_pos)
        self.derror_pos_std = tf.math.reduce_std(self.derror_pos)
        self.derror_pos_95ptl = 0 if len(self.derror_pos) == 0 else np.percentile(self.derror_pos, 95)
        self.derror_mean = tf.reduce_mean(self.derror_all)
        self.derror_std = tf.math.reduce_std(self.derror_all)
        self.derror_95ptl = 0 if len(self.derror_all) == 0 else np.percentile(self.derror_all, 95)

    def reset_states(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.fscore = 0
        self.iou = 0
        self.derror_pos = []
        self.derror_all = []
        self.derror_pos_mean = 0
        self.derror_pos_std = 0
        self.derror_pos_95ptl = 0
        self.derror_mean = 0
        self.derror_std = 0
        self.derror_95ptl = 0

    def result(self):
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'precision': self.precision,
            'recall': self.recall,
            'fscore': self.fscore,
            'iou': self.iou,
            # 'derror_pos': self.derror_pos,
            # 'derror_all': self.derror_all,
            'derror_pos_mean': tf.identity(self.derror_pos_mean),
            'derror_pos_std': tf.identity(self.derror_pos_std),
            'derror_pos_95ptl': tf.identity(self.derror_pos_95ptl),
            'derror_mean': tf.identity(self.derror_mean),
            'derror_std': tf.identity(self.derror_std),
            'derror_95ptl': tf.identity(self.derror_95ptl),

        }


class ConfMatrix(tf.keras.metrics.Metric):
    """
    Metrics class that computes the elements of the confusion matrix for the number of classes in the model.
    Other metrics are derived from this class

    Parameters
    ----------
    nclasses : int
        Number of classes
    is_logits : bool
        Whether the output of the model is in logits (True) or sigmoid (False)
    track_metric : str
        Metric to track by the class
    layer : int
        Layer where the metrics are calculated. -1 for output
    id_class : int or None
        ID of the class being evaluated, None for all classes
    name : str or None
        Name of the model
    eps : float
        Small number to avoid numerical errors in the calculation of the metrics
    kwargs : key, value mappings
        Other keyword arguments are passed through to class:`tf.keras.metrics.Metric`
    """

    def __init__(
            self,
            nclasses,
            is_logits,
            track_metric='confmat',
            layer=-1,
            id_class=None,
            name=None,
            eps=1e-10,
            **kwargs):

        self.nclasses = nclasses
        self.eps = eps
        self.is_logits = is_logits

        self.layer = layer
        self.track_metric = track_metric
        self.id_class = id_class
        if name is None:
            name = self.track_metric
            if self.id_class is not None:
                classname = str(self.id_class) if self.id_class < self.nclasses else 'mean'
                name += '_label' + str(classname)
            if self.layer is not None and self.layer > 0:
                name += '_layer' + str(self.layer)
        super(ConfMatrix, self).__init__(name=name, **kwargs)
        self.confmat = self.add_weight(name='ConfusionMatrix', shape=(self.nclasses, self.nclasses),
                                       initializer='zeros')
        self.precision = self.add_weight(name='Precision', shape=self.nclasses + 1, initializer='zeros')
        self.recall = self.add_weight(name='Recall', shape=self.nclasses + 1, initializer='zeros')
        self.fscore = self.add_weight(name='Fscore', shape=self.nclasses + 1, initializer='zeros')
        self.iou = self.add_weight(name='IoU', shape=self.nclasses + 1, initializer='zeros')

    # @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # if self.layer == 0:
        #     logits = y_pred[0]
        if self.layer == 0:
            logits = y_pred[0]
        elif self.layer > 0:
            logits = tf.gather(y_pred[1], self.layer - 1)
        else:
            logits = y_pred
        if isinstance(logits, dict):
            logits = logits['logits']
        elif isinstance(logits, (list, tuple)):
            logits = logits[0]
        if sample_weight is not None:
            raise Exception("sample_weight not implemented in this function")
        labels = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
        if self.is_logits:
            ypred = tf.nn.softmax(logits, axis=-1)
        else:
            ypred = logits
        ypred = tf.reshape(tf.argmax(ypred, axis=-1), [-1])
        conf_matrix = tf.math.confusion_matrix(labels, ypred, num_classes=self.nclasses, dtype=tf.float32)
        self.confmat.assign_add(conf_matrix)
        for label in range(self.nclasses):
            precision = self.confmat[label, label] / (tf.reduce_sum(self.confmat[:, label]) + self.eps)
            recall = self.confmat[label, label] / (tf.reduce_sum(self.confmat[label, :]) + self.eps)
            fscore = 2 * precision * recall / (precision + recall + eps)
            iou = fscore / (2 - fscore)
            self.precision[label].assign(precision)
            self.recall[label].assign(recall)
            self.fscore[label].assign(fscore)
            self.iou[label].assign(iou)
        self.precision[-1].assign(tf.reduce_mean(self.precision[:-1]))
        self.recall[-1].assign(tf.reduce_mean(self.recall[:-1]))
        self.fscore[-1].assign(tf.reduce_mean(self.fscore[:-1]))
        self.iou[-1].assign(tf.reduce_mean(self.iou[:-1]))

    def result(self):
        if self.track_metric is 'confmat':
            track_metric = tf.identity(self.confmat)
        elif self.track_metric is 'Fscore':
            track_metric = tf.identity(self.fscore)
        elif self.track_metric is 'Precision':
            track_metric = tf.identity(self.precision)
        elif self.track_metric is 'Recall':
            track_metric = tf.identity(self.recall)
        elif self.track_metric is 'IoU':
            track_metric = tf.identity(self.iou)
        else:
            raise Exception("Track metric unknown")
        if self.id_class is not None:
            track_metric = tf.gather(track_metric, self.id_class)
        return track_metric

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.confmat.assign(tf.zeros_like(self.confmat))
        self.precision.assign(tf.zeros_like(self.precision))
        self.recall.assign(tf.zeros_like(self.recall))
        self.fscore.assign(tf.zeros_like(self.fscore))
        self.iou.assign(tf.zeros_like(self.iou))


class IoU(ConfMatrix):
    """
    Intersection over Union, calculated from the confusion matrix
    """

    def __init__(self, nclasses, is_logits, name='IoU', id_class=None, **kwargs):
        self.id_class = id_class
        super(IoU, self).__init__(nclasses, is_logits, name=name, **kwargs)

    def result(self):
        track_metric = self.iou
        if self.id_class is not None:
            track_metric = tf.gather(track_metric, self.id_class)
        return tf.identity(track_metric)


class Precision(ConfMatrix):
    """
    Precision, calculated from the confusion matrix
    """

    def __init__(self, nclasses, is_logits, name='Precision', id_class=None, **kwargs):
        self.id_class = id_class
        super(Precision, self).__init__(nclasses, is_logits, name=name, **kwargs)

    def result(self):
        track_metric = self.precision
        if self.id_class is not None:
            track_metric = tf.gather(track_metric, self.id_class)
        return tf.identity(track_metric)


class Recall(ConfMatrix):
    """
    Recall, calculated from the confusion matrix
    """

    def __init__(self, nclasses, is_logits, name='Recall', id_class=None, **kwargs):
        self.id_class = id_class
        super(Recall, self).__init__(nclasses, is_logits, name=name, **kwargs)

    def result(self):
        track_metric = self.recall
        if self.id_class is not None:
            track_metric = tf.gather(track_metric, self.id_class)
        return tf.identity(track_metric)


class Fscore(ConfMatrix):
    """
    Fscore, calculated from the confusion matrix
    """

    def __init__(self, nclasses, is_logits, name='Fscore', id_class=None, **kwargs):
        self.id_class = id_class
        super(Fscore, self).__init__(nclasses, is_logits, name=name, **kwargs)

    def result(self):
        track_metric = self.fscore
        if self.id_class is not None:
            track_metric = tf.gather(track_metric, self.id_class)
        return tf.identity(track_metric)


def get_confmetrics(nclasses, is_logits, names=None, layer=-1, do_idclass=False):
    """
    Parses the arguments to return the appropriate metric

    Parameters
    ----------
    nclasses : int
        Number of classes to be accounted for in the metric
    is_logits : bool
        Whether the output of the model is in logits (True) or sigmoid (False)
    names : list (of str)
        List of names assigned to the metrics
    layer : int
        Layer where the metrics are calculated. -1 for output layer
    do_idclass : bool
        Whether each class should be accounted for individually in the metrics (True) or not (False)

    Returns
    -------
    tf.keras.metrics.Metric
        Metric class
    """
    metrics = [] * 4
    if names is None:
        names = [None] * 4
    for nm, mname in enumerate(['Fscore', 'Precision', 'Recall', 'IoU']):
        # mname_aux = mname if layer<1 else mname + '_layer' + str(layer)
        if do_idclass:
            for nclass in range(nclasses + 1):
                classname = str(nclass) if nclass < nclasses else 'mean'
                # name = mname_aux + '_label' + classname
                metrics.append(
                    ConfMatrix(
                        nclasses=nclasses,
                        is_logits=is_logits,
                        track_metric=mname,
                        name=None,
                        id_class=nclass,
                        layer=layer
                    )
                )
        else:
            metrics.append(
                ConfMatrix(
                    nclasses=nclasses,
                    is_logits=is_logits,
                    track_metric=mname,
                    name=names[nm],
                    id_class=None,
                    layer=layer
                )
            )
    return metrics
