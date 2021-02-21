# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import tensorflow as tf
import numpy as np
import datetime
import os
import warnings
import logging
import time
from utils import cnn_utils, common_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# class MetricsCallback(tf.keras.callbacks.Callback):
#
#     def __init__(self, metrics, datasets):

def basic_callback(model_dir, best_metric='val_loss', mode='auto', best=None):
    log_dir = os.path.join(model_dir, 'tensorboard', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model_best.tf'),
            monitor=best_metric,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            save_freq='epoch'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'model.tf'),
            monitor=best_metric,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode='min',
            save_freq='epoch'
        ),
        # Tensorboard
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            # histogram_freq=5,
            write_graph=True,
            update_freq='epoch',
        ),
        TimeCallback(log_dir),
        EpochCallback(model_dir),
        EpochBestCallback(model_dir, best_metric, mode=mode, best=best)
    ]


class TimeCallback(tf.keras.callbacks.Callback):

    def __init__(self, logdir):
        super(TimeCallback, self).__init__()
        self.time_batch = None
        self.time_epoch = None
        self.file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'time'))
        self.batch_count = -1

    def on_batch_begin(self, batch, logs=None):
        self.time_batch = time.time()

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        with self.file_writer.as_default():
            tf.summary.scalar('time_batch', time.time() - self.time_batch, step=self.batch_count)

    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer.as_default():
            tf.summary.scalar('time_epoch', time.time() - self.time_epoch, step=epoch)
        self.file_writer.flush()

    def on_train_end(self, logs=None):
        # To avoid residual summaries - not sure if it works
        self.file_writer.flush()


class EpochBestCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_dir, monitor, mode='auto', best=None):
        super(EpochBestCallback, self).__init__()
        self.model_dir = model_dir
        self.monitor = monitor
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = best or np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = best or -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = best or -np.Inf
            else:
                self.monitor_op = np.less
                self.best = best or np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            with open(os.path.join(self.model_dir, 'last_epoch_best.txt'), 'w') as f:
                f.write('epoch {} -- value {}'.format(epoch, self.best))


class EpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir
        super(EpochCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        with open(os.path.join(self.model_dir, 'last_epoch.txt'), 'w') as f:
            f.write(str(epoch))


class LayersCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_dir, ds_vis, model, layer_names=None, n_vischannels=5, update_freq=10, auxname='',
                 do_ms=None):
        super(LayersCallback, self).__init__()

        self.do_ms = do_ms
        self.auxname = auxname
        self.feat_extraction_model = model or self.create_model()
        if self.do_ms:
            self.nmarkers = self.feat_extraction_model.input.shape[-1]
            self.mcomb = common_utils.marker_combinations(self.nmarkers)
            if isinstance(self.do_ms, int):
                ncomb = min(self.do_ms, len(self.mcomb))
                mcomb_ind = (np.random.rand(ncomb) * len(self.mcomb)).astype(np.int)
                self.mcomb = np.array(self.mcomb)[mcomb_ind]
            elif self.nmarkers == 5:
                self.mcomb = np.array([
                    (0, 1, 3),
                    (1,),
                    (3, 4),
                    (2, 3),
                    (1, 2),
                    (0,)
                ])
            # else:
            #     ncomb = len(self.mcomb)
            #     mcomb_ind = (np.random.rand(ncomb) * len(self.mcomb)).astype(np.int)
            #     self.mcomb = np.array(self.mcomb)[mcomb_ind]
            self.file_writer = {
                m: tf.summary.create_file_writer(os.path.join(model_dir, 'tensorboard', 'images_' + str(m)))
                for m in self.mcomb
            }
        else:
            self.file_writer = tf.summary.create_file_writer(
                os.path.join(model_dir, 'tensorboard',
                             'images_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.ds_vis = ds_vis
        self.layer_names = layer_names
        self.n_vischannels = n_vischannels
        self.update_freq = update_freq
        self.timing = 0.

    def create_model(self):
        layers = [layer.output for layer in self.model.layers if
                  ((self.layer_names is None) or (layer.name in self.layer_names))]
        return tf.keras.Model(inputs=self.model.input, outputs=layers)

    def on_epoch_end(self, epoch, logs=None):
        # Use only every self.update_freq epochs
        tstart = time.time()
        if epoch % self.update_freq:
            return

        for data_batch in self.ds_vis:
            x, _ = cnn_utils.split_batch(data_batch)
            if self.do_ms:
                for m in self.mcomb:
                    lsh = x.shape[0:-1]
                    mask = tf.stack(
                        [tf.cond(x in tuple(m), lambda: tf.ones(lsh), lambda: tf.zeros(lsh)) for x in
                         range(self.nmarkers)],
                        axis=-1)
                    xms = x * mask
                    self.visualize_epoch(xms, epoch, mname=m)
            else:
                self.visualize_epoch(x, epoch)
            break

        epoch_time = time.time() - tstart
        logger.debug("Writing activations in this epoch took {} seconds".format(epoch_time))
        self.timing += epoch_time

        return

    def visualize_epoch(self, x, epoch, mname=None):
        is_3d = len(self.feat_extraction_model.input.shape) == 5
        feats = self.feat_extraction_model(x)
        if len(self.feat_extraction_model.outputs) == 1:
            feats = [feats]
        for cf, feat in enumerate(feats):
            # if cf == 0 and epoch > 0:
            #     continue
            # norm = True if epoch == 0 else False
            lname = self.auxname + self.feat_extraction_model.output[cf].name
            ndims = feat.ndim
            if ndims == 4:
                feat = tf.expand_dims(feat, axis=-1)
            if is_3d:
                feat_channels = feat.shape[-1]
                nchannels = feat_channels if (
                        self.n_vischannels is None
                ) else min(self.n_vischannels, feat_channels)
                lchannels = np.around(np.linspace(0, feat.shape[-2] - 1, nchannels)).astype(np.int)
                self.vis_image(feat[:, feat.shape[1] // 2, ...], feat_channels, lname + '_xy', epoch, mname=mname)
                self.vis_image(feat[:, :, feat.shape[2] // 2, ...], feat_channels, lname + '_xz', epoch, mname=mname)
                self.vis_image(feat[:, :, :, feat.shape[3] // 2, ...], feat_channels, lname + '_yz', epoch, mname=mname)
            else:
                for nmod in range(feat.shape[-1]):
                    if ndims == 5:
                        lcname = lname + '_mod' + str(nmod)
                    else:
                        lcname = lname
                    feat_channels = feat.shape[-2]
                    self.vis_image(feat[..., nmod], feat_channels, lcname, epoch, mname=mname)

    def vis_image(self, feat, feat_channels, lcname, epoch, mname=None):
        nchannels = feat_channels if (self.n_vischannels is None) else min(self.n_vischannels,
                                                                           feat_channels)
        lchannels = np.around(np.linspace(0, feat.shape[-1] - 1, nchannels)).astype(np.int)
        for channel in lchannels:
            img_aux = feat[..., channel]
            img = np.reshape(img_aux, [-1] + img_aux.shape[1::].as_list() + [1])
            # if norm:
            #     img /= 255.0
            fmin = tf.reduce_min(feat)
            fmax = tf.reduce_max(feat)
            img = (img - fmin) / (fmax - fmin)
            if mname:
                file_writer = self.file_writer[mname]
            else:
                file_writer = self.file_writer
            with file_writer.as_default():
                tf.summary.image(name=lcname + '_C' + str(channel),
                                 data=img,
                                 step=epoch)
            file_writer.flush()

    def on_train_end(self, logs=None):
        logger.debug("Writing activations for all epochs took {} seconds".format(self.timing))
