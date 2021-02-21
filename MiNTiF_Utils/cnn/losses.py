# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DiceLoss(tf.losses.Loss):
    """
    Dice loss

    References
    ----------
    .. Adapted from https://github.com/baumgach/acdc_segmenter
    """

    def __init__(self, epsilon=1e-10, use_hard_pred=False, **kwargs):
        super().__init__()
        self.only_foreground = kwargs.get('only_foreground', False)
        self.epsilon = epsilon
        self.use_hard_pred = use_hard_pred
        mode = kwargs.get('mode', None)
        if mode == 'macro':
            self.sum_over_labels = False
            self.sum_over_batches = False
        elif mode == 'macro_robust':
            self.sum_over_labels = False
            self.sum_over_batches = True
        elif mode == 'micro':
            self.sum_over_labels = True
            self.sum_over_batches = False
        elif mode is None:
            self.sum_over_labels = kwargs.get('per_structure')  # Intentionally no default value
            self.sum_over_batches = kwargs.get('sum_over_batches', False)
        else:
            raise ValueError("Encountered unexpected 'mode' in dice_loss: '%s'" % mode)

    def call(self, labels, logits):
        dice_per_img_per_lab = self.get_dice(logits=logits, labels=labels)
        if self.only_foreground:
            if self.sum_over_batches:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[:-1])
            else:
                loss = 1 - tf.reduce_mean(dice_per_img_per_lab[:, :-1])
        else:
            loss = 1 - tf.reduce_mean(dice_per_img_per_lab)
        return loss

    def get_dice(self, labels, logits):
        ndims = logits.get_shape().ndims
        prediction = logits
        # prediction = tf.nn.softmax(logits)
        if self.use_hard_pred:
            # This casts the predictions to binary 0 or 1
            prediction = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])

        intersection = tf.multiply(prediction, labels)

        if ndims == 5:
            reduction_axes = [1, 2, 3]
        else:
            reduction_axes = [1, 2]

        if self.sum_over_batches:
            reduction_axes = [0] + reduction_axes

        if self.sum_over_labels:
            reduction_axes += [reduction_axes[-1] + 1]  # also sum over the last axis

        # Reduce the maps over all dimensions except the batch and the label index
        i = tf.reduce_sum(intersection, axis=reduction_axes)
        l = tf.reduce_sum(prediction, axis=reduction_axes)
        r = tf.reduce_sum(labels, axis=reduction_axes)

        dice_per_img_per_lab = 2 * i / (l + r + self.epsilon)

        return dice_per_img_per_lab


class CrossEntropyWeighted(tf.losses.Loss):
    """
    Weighted cross entropy loss with logits
    """

    def __init__(self, class_weights, key_out=None, name='CrossEntropyWeigthed', **kwargs):
        super(CrossEntropyWeighted, self).__init__(name=name, **kwargs)
        self.class_weights = class_weights
        self.n_class = len(self.class_weights)
        self.key_out = key_out

    @tf.function
    def call(self, labels, logits):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])

        class_weights = tf.constant(np.array(self.class_weights, dtype=np.float32))

        weight_map = tf.multiply(flat_labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)

        return loss


def get_loss(name_loss, class_weights=None):
    """
    Parses the parameters to get the correct loss

    Parameters
    ----------
    name_loss : str
        Name of the loss function to use
    class_weights : list (of floats) or None
        Weights of the different classes employed

    Returns
    -------
    tf.losses.Loss
        Loss function as required by a tf model
    """

    if name_loss == 'dice':
        return DiceLoss(mode='macro_robust', only_foreground=True)
    elif name_loss == 'crossentropy':
        if class_weights is None:
            raise Exception("class_weights should be declared for weighted cross entropy")
        return CrossEntropyWeighted(class_weights)
    else:
        raise Exception("The loss {} has not been implemented".format(name_loss))
