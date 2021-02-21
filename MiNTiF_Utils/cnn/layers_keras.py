# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
import tensorflow as tf
from utils.cnn_utils import calc_layercrop
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StatsAbstraction(tf.keras.layers.Layer):
    """
    Statistical moments in the abstraction layer for HeMIS
    """

    def __init__(self, **kwargs):
        super(StatsAbstraction, self).__init__(**kwargs)
        self.lmean = tf.math.reduce_mean
        self.lstd = tf.math.reduce_std
        self.lconc = tf.keras.layers.Concatenate(axis=-1)

    @tf.function
    def call(self, inputs, **kwargs):
        # mlayer = self.lmean(inputs, axis=-1)
        # stdlayer = self.lstd(inputs, axis=-1)
        mlayer, stdlayer = tf.nn.moments(inputs, axes=-1, name='calculate_moments')
        return self.lconc([mlayer, stdlayer])


def crop_match(x, y):
    ref_size = y.shape[1:-1]
    sizediff = tf.cast(tf.divide(tf.math.subtract(x.shape[1:-1], ref_size), 2), tf.int32)
    # sizediff = (x.shape[1:-1] - ref_size)/2
    init_crop = tf.stack([0, sizediff[0], sizediff[1], 0])
    ref_size_fix = tf.stack([-1, ref_size[0], ref_size[1], x.shape[-1]])
    # return tf.slice(x, init_crop, ref_size_fix.numpy())
    return tf.slice(x, init_crop, ref_size_fix)


def upsampling_custom(scale_factor, ndims=2, nodes=None, name=None, interpolation='nearest'):
    result = tf.keras.Sequential(name=name)
    if ndims == 2:
        result.add(tf.keras.layers.UpSampling2D(
            size=scale_factor,
            interpolation=interpolation))
        if nodes is not None:
            result.add(
                tf.keras.layers.Conv2D(nodes, [2, 2], padding='same')
            )
    elif ndims == 3:
        result.add(tf.keras.layers.UpSampling3D(
            size=scale_factor))
        if nodes is not None:
            result.add(
                tf.keras.layers.Conv3D(nodes, [2, 2, 2], padding='same')
            )
    else:
        logger.error("Method not implemented for {} dimensions".format(ndims))
    return result


def batchnorm_bool(do_batchnorm, name=None):
    result = tf.keras.Sequential(name=name)
    if do_batchnorm:
        result.add(
            tf.keras.layers.BatchNormalization(
                axis=-1,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones'))
    return result


def blockconv(nodes, do_batchnorm, ksize, padding, activation='relu', activation2='relu', name=None, ndims=2, rdrop=0):
    if ndims == 2:
        result = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=nodes,
                    kernel_size=ksize,
                    strides=(1, 1),
                    padding=padding,
                    dilation_rate=(1, 1),
                    activation=activation
                ),
                tf.keras.layers.Conv2D(
                    filters=nodes,
                    kernel_size=ksize,
                    strides=(1, 1),
                    padding=padding,
                    dilation_rate=(1, 1),
                    activation=activation2
                )],
            name=name)
    elif ndims == 3:
        result = tf.keras.Sequential(
            [
                tf.keras.layers.Conv3D(
                    filters=nodes,
                    kernel_size=ksize,
                    strides=(1, 1, 1),
                    padding=padding,
                    dilation_rate=(1, 1, 1),
                    activation=activation
                ),
                tf.keras.layers.Conv3D(
                    filters=nodes,
                    kernel_size=ksize,
                    strides=(1, 1, 1),
                    padding=padding,
                    dilation_rate=(1, 1, 1),
                    activation=activation2
                )],
            name=name)
    else:
        logger.error("Method not implemented for {} dimensions".format(ndims))
    if do_batchnorm:
        result.add(
            batchnorm_bool(do_batchnorm)
        )
    if rdrop > 0:
        result.add(
            tf.keras.layers.Dropout(rdrop)
        )
    return result


def skipconnect(level, nlevels, scale_factor, padding=False, name=None, ndims=2):
    if padding == 'same':
        cropind = 0
    else:
        cropind = [int(a) for a in calc_layercrop(level, nlevels, scale_factor)]
    if ndims == 2:
        result = tf.keras.Sequential(
            [tf.keras.layers.Cropping2D(cropind)],
            name=name)
    elif ndims == 3:
        result = tf.keras.Sequential(
            [tf.keras.layers.Cropping3D(cropind)],
            name=name)
    else:
        logger.error("Method not implemented for {} dimensions".format(ndims))
    return result


class GetMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GetMask, self).__init__(**kwargs)
        self.lmask = tf.keras.layers.Lambda(lambda x: tf.reduce_all(tf.equal(tf.zeros_like(x), x), axis=[0, 1, 2]))
        self.inv_mask = tf.keras.layers.Lambda(lambda x: tf.logical_not(x))

    def call(self, inputs, **kwargs):
        nmask = self.lmask(inputs)
        return self.inv_mask(nmask)


class ChannelDropout(tf.keras.layers.Layer):
    """
    Marker Sampling strategy
    """

    def __init__(self, drop_rate=0.5, **kwargs):
        super(ChannelDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        self.lmaindrop = tf.keras.layers.Lambda(lambda x: tf.random.uniform([x.shape[-1]]) < self.drop_rate)
        self.backupdrop = tf.keras.layers.Lambda(
            lambda x: tf.one_hot(
                tf.random.uniform([1], minval=0, maxval=x.shape[-1], dtype=tf.dtypes.int32)[0],
                x.shape[-1]) > 0.5
        )
        self.finaldrop = tf.keras.layers.Lambda(
            lambda x: tf.cond(tf.reduce_any(x[0]), lambda: x[0], lambda: x[1])
        )

    def call(self, inputs, training=None):
        x_in, x_out = inputs
        if training:
            drop_main = self.lmaindrop(x_in)
            # To avoid 0 channels
            drop_backup = self.backupdrop(x_in)
            drop_channels = self.finaldrop((drop_main, drop_backup))
            x_in = tf.boolean_mask(x_in, drop_channels, axis=3)
            x_out = tf.boolean_mask(x_out, drop_channels, axis=4)
        return x_in, x_out


class MaskChannels(tf.keras.layers.Layer):
    """
    Mask channels for their sampling
    """

    def __init__(self, do_dropout=False, **kwargs):
        super(MaskChannels, self).__init__(**kwargs)
        self.drop_rate = 0.85
        self.ldropout = ChannelDropout()
        self.cmask = GetMask()
        self.boolmask = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1], axis=4))
        self.do_dropout = do_dropout

    @tf.function
    def call(self, inputs, **kwargs):
        x_inaux, x_outaux = inputs
        if self.do_dropout:
            x_in, x_out = self.ldropout((x_inaux, x_outaux), **kwargs)
        else:
            x_in, x_out = x_inaux, x_outaux
        cmask = self.cmask(x_in)
        return self.boolmask((x_out, cmask))

    def get_config(self):
        config = super(MaskChannels, self).get_config()
        config.update({
            "do_dropout": self.do_dropout
        })


class DeleteOutput(tf.keras.layers.Layer):
    def __init__(self, ch_ind, **kwargs):
        super(DeleteOutput, self).__init__(**kwargs)
        self.ch_ind = ch_ind
        self.iszero = tf.keras.layers.Lambda(lambda x: tf.reduce_all(tf.equal(tf.zeros_like(x), x)))

    @tf.function
    def call(self, inputs, **kwargs):
        x_in_aux, x_out = inputs
        x_in = tf.gather(x_in_aux, self.ch_ind, axis=-1)
        return tf.cond(self.iszero(x_in), lambda: -1. * tf.ones_like(x_out), lambda: x_out)

    def get_config(self):
        config = super(DeleteOutput, self).get_config()
        config.update({
            "ch_ind": self.ch_ind
        })


class AdaptiveCrop(tf.keras.layers.Layer):
    def __init__(self, expected_size, name='AdaptiveCrop', **kwargs):
        super().__init__(name=name, **kwargs)
        self.expected_size = expected_size

    def build(self, input_shape):
        super(AdaptiveCrop, self).build(input_shape)
        cropsize = tf.subtract(input_shape[1:-1], self.expected_size)
        assert tf.reduce_all(cropsize >= 0)

        cropinds = np.array([
            cropsize // 2,
            cropsize // 2 + cropsize % 2
        ]).T
        self.croplayer = tf.keras.layers.Cropping2D(cropping=cropinds)

    @tf.function
    def call(self, inputs, **kwargs):
        return self.croplayer(inputs)

    def get_config(self):
        config = super(AdaptiveCrop, self).get_config()
        config.update({
            'expected_size': self.expected_size
        })


def adaptive_crop(X, expected_size, name=None):
    cropsize = np.array(X.shape[1:-1]) - np.array(expected_size)
    if (np.array(cropsize) > 0).any():
        cropinds = np.array([
            cropsize // 2,
            cropsize // 2 + cropsize % 2
        ]).T
        X = tf.keras.layers.Cropping2D(cropping=cropinds, name=name)(X)
    elif (np.array(cropsize) < 0).any():
        raise Exception("This case needs to be implemented")
    return X


class ChannelGather(tf.keras.layers.Layer):
    """
    Layer to gather a designed channel
    """

    def __init__(self, channel, **kwargs):
        super(ChannelGather, self).__init__(**kwargs)
        self.channel = channel
        self.gather = tf.keras.layers.Lambda(lambda x: tf.gather(x, indices=self.channel, axis=-1), name='gather')
        self.expand = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), name='expand')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.gather(inputs)
        return self.expand(x)

    def get_config(self):
        config = super(ChannelGather, self).get_config()
        config.update({
            'channel': self.channel
        })


class ChannelsConcatenate(tf.keras.layers.Layer):
    """
    Merge channels that have been processed in different blocks
    """

    def __init__(self, **kwargs):
        super(ChannelsConcatenate, self).__init__(**kwargs)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

    @tf.function
    def call(self, inputs, **kwargs):
        Yb_list = [None] * len(inputs)
        for nm, Xm in enumerate(inputs):
            Yb_list[nm] = tf.expand_dims(Xm, axis=-1)
        Yb = self.concatenate(Yb_list)
        return Yb


class StackConcatenate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StackConcatenate, self).__init__(**kwargs)
        self.conc = tf.keras.layers.Concatenate(axis=0)

    @tf.function
    def call(self, inputs, **kwargs):
        x1, x2 = inputs
        s1 = tf.expand_dims(x1, 0)
        s2 = tf.stack(x2)
        return self.conc([s1, s2])


class SEblock(tf.keras.layers.Layer):
    """
    Squeeze and Excitation block

    References
    ----------
    .. [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, *Squeeze-and-Excitation Networks*, CVPR 2018
    """

    def __init__(self, cratio, **kwargs):
        super(SEblock, self).__init__(**kwargs)
        self.cratio = cratio
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        nchannels = input_shape[-1]
        if self.cratio > nchannels:
            raise Exception("The ratio of compression cannot be greater than the number of channels")
        self.dense1 = tf.keras.layers.Dense(nchannels // self.cratio, activation='relu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(nchannels, activation='sigmoid', use_bias=False)

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.globalpool(inputs)
        x = self.dense1(x)
        excitation = self.dense2(x)
        output = tf.multiply(inputs, excitation)
        return output

    def get_config(self):
        config = super(SEblock, self).get_config()
        config.update({
            'ratio': self.cratio
        })


class SEdomainDom(tf.keras.layers.Layer):
    """
    Cross-domain attention block adapted to different independent modules

    References
    ----------
    .. [2] Xudong Wang, Zhaowei Cai, Dashan Gao, Nuno Vasconcelos, *Towards Universal Object Detection by Domain
    Attention*, CVPR 2019
    """

    def __init__(self, cratio=2, nmods=3, dom_assign_act='sigmoid', **kwargs):
        super(SEdomainDom, self).__init__(**kwargs)
        if dom_assign_act == 'softmax':
            self.dom_assign_act = tf.keras.activations.softmax
        elif dom_assign_act == 'sigmoid':
            self.dom_assign_act = tf.keras.activations.sigmoid
        else:
            raise Exception("dom_assign_act unknown: {}".format(dom_assign_act))

        self.cratio = cratio
        self.nmods = nmods
        self.lsqueeze = tf.keras.layers.GlobalAveragePooling3D()
        self.lconcat = tf.keras.layers.Concatenate(axis=0)
        self.dom_assign = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(self.nmods),
            tf.keras.layers.Layer(lambda x: self.dom_assign_act(x))
        ])
        self.lsigm = tf.keras.activations.sigmoid

    def build(self, input_shape):
        super(SEdomainDom, self).build(input_shape)
        self.nfeats = input_shape[-1]
        self.nodes1 = self.nfeats // self.cratio
        self.adapters = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(self.nodes1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.nfeats)
            ], name='AdapterM' + str(nm))
            for nm in range(self.nmods)]

    @tf.function
    def call(self, inputs, **kwargs):
        xsq = self.lsqueeze(inputs)
        Xmods_arr = tf.TensorArray(
            size=self.nmods,
            dynamic_size=False,
            clear_after_read=True,
            infer_shape=True,
            dtype=tf.float32,
            name='array_mods'
        )
        for nm, adaptmod in enumerate(self.adapters):
            Xmods_arr.write(nm, adaptmod(xsq))
        Xmods_aux = Xmods_arr.stack()
        Xmods_aux.set_shape([self.nmods, Xmods_aux.shape[1], self.nfeats])
        xmods = tf.transpose(Xmods_aux, [1, 2, 0])
        # xmods = self.lconcat(Xmods_aux)
        xassign = self.dom_assign(inputs)
        xscale = tf.matmul(xmods, tf.expand_dims(xassign, 1), transpose_b=True)
        xscale = tf.gather(xscale, 0, axis=-1)
        xscale_sigm = self.lsigm(xscale)
        xscale_re = tf.reshape(xscale_sigm, [-1, 1, 1, 1, self.nfeats])
        xexc = tf.multiply(inputs, xscale_re)
        return xexc


class SEdomain(tf.keras.layers.Layer):
    """
        Cross-domain attention block

        References
        ----------
        .. [2] Xudong Wang, Zhaowei Cai, Dashan Gao, Nuno Vasconcelos, *Towards Universal Object Detection by Domain
        Attention*, CVPR 2019
        """

    def __init__(self, cratio=2, nmods=3, dom_assign_act='softmax', **kwargs):
        super(SEdomain, self).__init__(**kwargs)
        if dom_assign_act == 'softmax':
            self.dom_assign_act = tf.keras.activations.softmax
        elif dom_assign_act == 'sigmoid':
            self.dom_assign_act = tf.keras.activations.sigmoid
        else:
            raise Exception("dom_assign_act unknown: {}".format(dom_assign_act))

        self.cratio = cratio
        self.nmods = nmods
        self.lsqueeze = tf.keras.layers.GlobalAveragePooling2D()
        self.lconcat = tf.keras.layers.Concatenate(axis=0)
        self.dom_assign = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.nmods),
            tf.keras.layers.Layer(lambda x: self.dom_assign_act(x))
        ])
        self.lsigm = tf.keras.activations.sigmoid

    def build(self, input_shape):
        super(SEdomain, self).build(input_shape)
        self.nfeats = input_shape[-1]
        self.nodes1 = self.nfeats // self.cratio
        self.adapters = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(self.nodes1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.nfeats)
            ], name='AdapterM' + str(nm))
            for nm in range(self.nmods)]

    @tf.function
    def call(self, inputs, **kwargs):
        xsq = self.lsqueeze(inputs)
        Xmods_arr = tf.TensorArray(
            size=self.nmods,
            dynamic_size=False,
            clear_after_read=True,
            infer_shape=True,
            dtype=tf.float32,
            name='array_mods'
        )
        for nm, adaptmod in enumerate(self.adapters):
            Xmods_arr.write(nm, adaptmod(xsq))
        Xmods_aux = Xmods_arr.stack()
        Xmods_aux.set_shape([self.nmods, Xmods_aux.shape[1], self.nfeats])
        xmods = tf.transpose(Xmods_aux, [1, 2, 0])
        # xmods = self.lconcat(Xmods_aux)
        xassign = self.dom_assign(inputs)
        xscale = tf.matmul(xmods, tf.expand_dims(xassign, 1), transpose_b=True)
        xscale = tf.gather(xscale, 0, axis=-1)
        xscale_sigm = self.lsigm(xscale)
        xscale_re = tf.reshape(xscale_sigm, [-1, 1, 1, self.nfeats])
        xexc = tf.multiply(inputs, xscale_re)
        return xexc


class SEspatialMod(tf.keras.layers.Layer):
    """
    Spatial Squeeze and Excitation block adapted to different modules

    References
    ----------
    .. [3] Abhijit Guha Roy, Nassir Navab, Christian Wachinger, *Concurrent Spatial and Channel Squeeze & Excitation in
    Fully Convolutional Networks*, MICCAI 2018
    """

    def __init__(self, cratio=2, **kwargs):
        super(SEspatialMod, self).__init__(**kwargs)
        self.cratio = cratio

    def build(self, input_shape):
        super(SEspatialMod, self).build(input_shape)
        self.nchannels = input_shape[-1]
        self.squeeze = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-2))
        self.conv1 = tf.keras.layers.Conv2D(self.nchannels // self.cratio, (1, 1))
        self.conv2 = tf.keras.layers.Conv2D(self.nchannels, (1, 1))
        self.makemask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))

    def call(self, inputs, **kwargs):
        xsq = self.squeeze(inputs)
        x1 = self.conv1(xsq)
        x2 = self.conv2(x1)
        xex = self.makemask(x2)
        xout = tf.math.multiply(inputs, xex)
        return xout

    def get_config(self):
        config = super(SEspatialMod, self).get_config()
        config.update({
            'cratio': self.cratio
        })


class SEchannelsMod(tf.keras.layers.Layer):
    """
    Squeeze and Excitation block adapted to different modules

    References
    ----------
    .. [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, *Squeeze-and-Excitation Networks*, CVPR 2018
    """

    def __init__(self, cratio=2, **kwargs):
        super(SEchannelsMod, self).__init__(**kwargs)
        self.cratio = cratio
        self.globalpool = tf.keras.layers.GlobalAveragePooling3D()

    def build(self, input_shape):
        super(SEchannelsMod, self).build(input_shape)
        nchannels = input_shape[-1]
        self.dense1 = tf.keras.layers.Dense(nchannels // self.cratio, activation='relu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(nchannels, activation='sigmoid', use_bias=False)

    @tf.function
    def call(self, inputs, **kwargs):
        # SE block
        xg = self.globalpool(inputs)
        xsq = self.dense1(xg)
        excitation = self.dense2(xsq)
        excitation = tf.reshape(excitation, [-1, 1, 1, 1, excitation.shape[-1]])
        Y = tf.multiply(inputs, excitation)
        return Y

    def get_config(self):
        config = super(SEchannelsMod, self).get_config()
        config.update({
            'ratio': self.cratio
        })


class SEchannels(tf.keras.layers.Layer):
    """
    Squeeze and Excitation block adapted to work accross channels

    References
    ----------
    .. [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, *Squeeze-and-Excitation Networks*, CVPR 2018
    """

    def __init__(self, cratio=2, activation1='relu', activation2='sigmoid', do_bias=False, merge_op='multiply',
                 **kwargs):
        super(SEchannels, self).__init__(**kwargs)
        self.cratio = cratio
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.activation1 = activation1
        self.activation2 = activation2
        self.do_bias = do_bias
        self.merge_op = merge_op
        if self.merge_op == 'multiply':
            self.lmerge = tf.multiply
        elif self.merge_op == 'add':
            self.lmerge = tf.add
        else:
            raise Exception("merge_op {} unknown".format(merge_op))

    def build(self, input_shape):
        super(SEchannels, self).build(input_shape)
        nchannels = input_shape[-1]
        if self.cratio < 1:
            nnodes = 64
        else:
            nnodes = nchannels // self.cratio

        if self.activation1 == 'lrelu':
            self.lactivation1 = tf.keras.layers.LeakyReLU()
            act_aux1 = None
        else:
            act_aux1 = self.activation1
        if self.activation2 == 'lrelu':
            self.lactivation2 = tf.keras.layers.LeakyReLU()
            act_aux2 = None
        else:
            act_aux2 = self.activation2
        self.dense1 = tf.keras.layers.Dense(nnodes, activation=act_aux1, use_bias=self.do_bias)
        self.dense2 = tf.keras.layers.Dense(nchannels, activation=act_aux2, use_bias=self.do_bias)

    @tf.function
    def call(self, inputs, **kwargs):
        # SE block
        xg = self.globalpool(inputs)
        xsq = self.dense1(xg)
        if self.activation1 == 'lrelu':
            xsq = self.lactivation1(xsq)
        excitation = self.dense2(xsq)
        if self.activation2 == 'lrelu':
            excitation = self.lactivation2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, excitation.shape[-1]])
        Y = self.lmerge(inputs, excitation)
        return Y

    def get_config(self):
        config = super(SEchannels, self).get_config()
        config.update({
            'cratio': self.cratio,
            'activation1': self.activation1,
            'activation2': self.activation2,
            'do_bias': self.do_bias,
            'merge_op': self.merge_op
        })


class SEchannelsModMask(SEchannelsMod):
    def __init__(self, threshold=None, name='SEmask', **kwargs):
        super(SEchannelsModMask, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        if self.threshold:
            raise Exception("Not implemented")

    @tf.function
    def call(self, inputs, **kwargs):
        x_in, x_out_aux = inputs
        x_out = super(SEchannelsModMask, self).call(x_out_aux)

        # Masking
        nmask = tf.reduce_all(tf.equal(tf.zeros_like(x_in), x_in), axis=[0, 1, 2])
        cmask_in = tf.logical_not(nmask)

        Y = tf.boolean_mask(x_out, cmask_in, axis=4)
        return Y

    def get_config(self):
        config = super(SEchannelsModMask, self).get_config()
        config.update({
            'threshold': self.threshold
        })


class ChDrop_nonorm(tf.keras.layers.Layer):

    def __init__(self, rate=0.5, do_scale=False, do_train_scale=False, ndims=2, **kwargs):
        super(ChDrop_nonorm, self).__init__(**kwargs)
        self.ndims = ndims
        self.reduce_axes = [0, 1, 2, 3] if ndims == 3 else [0, 1, 2]
        self.do_train_scale = do_train_scale
        self.drop_rate = rate
        self.do_scale = do_scale
        if self.do_train_scale and self.do_scale:
            raise Exception("do_scale and do_train_scale cannot be both set to True")

    def build(self, input_shape):
        super(ChDrop_nonorm, self).build(input_shape)
        self.nchannels = input_shape[-1]
        self.reshape_axes = [-1, 1, 1, 1, self.nchannels] if self.ndims == 3 else [-1, 1, 1, self.nchannels]

    @tf.function
    def get_random_vector(self, xin):
        xred = tf.reduce_sum(xin, self.reduce_axes)
        ch_exist = tf.logical_not(tf.equal(xred, tf.zeros_like(xred)))
        if tf.reduce_any(ch_exist):
            def cond(x):
                channels_kept = tf.logical_and(ch_exist, tf.logical_not(x))
                any_channel = tf.reduce_any(channels_kept, axis=0)
                # We iterate when no channels
                return tf.logical_not(any_channel)

            get_mask = lambda: [tf.random.uniform([self.nchannels]) < self.drop_rate]
            xdrop = tf.while_loop(cond, lambda x: get_mask(), get_mask())
            # Mask the channels we keep (opposite of xdrop)
            mask_aux = tf.cast(tf.logical_not(xdrop), tf.float32)
        else:
            # logger.warning("NO CHANNELS AVAILABLE IN THIS SAMPLE!")
            mask_aux = tf.constant([0.] * 5, dtype=tf.float32)
        mask = tf.reshape(mask_aux, self.reshape_axes)
        return mask

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        if training:
            mask = self.get_random_vector(inputs)
            outputs = tf.multiply(inputs, mask)

        else:
            outputs = inputs
        nch_comb = tf.math.reduce_sum(
            tf.cast(tf.not_equal(tf.reduce_sum(outputs, axis=self.reduce_axes), 0), tf.float32))
        if self.do_train_scale and training:
            keep_prob = 1 - self.drop_rate
            scale = 1 / keep_prob
            outputs = outputs * scale
        if self.do_scale and tf.greater(nch_comb, 0):
            outputs = outputs / (nch_comb / self.nchannels)
        return outputs

    def get_config(self):
        config = super(ChDrop_nonorm, self).get_config()
        config.update({
            'drop_rate': self.drop_rate,
            'do_scale': self.do_scale
        })
        return config


class ChCombAtt(tf.keras.layers.Layer):
    """
    Marker Excite module
    """

    def __init__(self, do_bias=True, merge_op='multiply', post_activation='relu', **kwargs):
        super(ChCombAtt, self).__init__(**kwargs)
        self.do_bias = do_bias
        self.merge_op = merge_op
        if self.merge_op == 'multiply':
            self.lmerge = tf.multiply
        elif self.merge_op == 'add':
            self.lmerge = tf.add
        elif self.merge_op == 'both':
            pass
        else:
            raise Exception("merge_op {} unknown".format(merge_op))

    def build(self, input_shape):
        super(ChCombAtt, self).build(input_shape)
        self.nmods = input_shape[1][-1]
        self.nfeats = input_shape[0][-1]
        self.nnodes = 2 ** self.nmods - 1
        self.dense1 = tf.keras.layers.Dense(self.nnodes, activation='relu', use_bias=self.do_bias)
        self.dense2 = tf.keras.layers.Dense(self.nfeats, activation='sigmoid', use_bias=self.do_bias)
        if len(input_shape[0]) == 4:
            self.lreshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, x.shape[-1]]))
        elif len(input_shape[0]) == 5:
            self.lreshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, 1, x.shape[-1]]))
        else:
            raise Exception("function not implemented for this number of dimensions")

    @tf.function
    def call(self, inputs, **kwargs):
        x, x_chcomb = inputs
        # SE block
        x1 = self.dense1(x_chcomb)
        excitation = self.dense2(x1)
        excitation = self.lreshape(excitation)
        Y = self.lmerge(x, excitation)
        return Y


class ChCombAttPostAct(ChCombAtt):
    """
    Marker Excite module with activation after weighting
    """

    def __init__(self, **kwargs):
        super(ChCombAttPostAct, self).__init__(**kwargs)
        self.postact = tf.keras.layers.ReLU()

    def build(self, input_shape):
        super(ChCombAttPostAct, self).build(input_shape)
        self.dense2_bias = tf.keras.layers.Dense(self.nfeats, activation='relu')

    @tf.function
    def call(self, inputs, **kwargs):
        x, x_chcomb = inputs
        # SE block
        x1 = self.dense1(x_chcomb)
        excitation = self.dense2(x1)
        excitation_bias = self.dense2_bias(x1)
        Y = tf.multiply(x, self.lreshape(excitation)) + self.lreshape(excitation_bias)
        Yout = self.postact(Y)
        return Yout


class DomChCombAtt(tf.keras.layers.Layer):

    def __init__(self, cratio=2, ndoms=3, cust_domassign=0, dom_assign_act='sigmoid', **kwargs):
        super(DomChCombAtt, self).__init__(**kwargs)
        self.cust_domassign = cust_domassign
        if dom_assign_act == 'softmax':
            self.dom_assign_act = lambda x: tf.keras.activations.softmax(x)
        elif dom_assign_act == 'sigmoid':
            self.dom_assign_act = lambda x: tf.keras.activations.sigmoid(x)
        elif dom_assign_act == 'relu1':
            self.dom_assign_act = lambda x: tf.keras.activations.relu(x) + 1
        elif dom_assign_act == 'tanh':
            self.dom_assign_act = lambda x: tf.keras.activations.tanh(x)
        elif dom_assign_act is None:
            self.dom_assign_act = lambda x: x
        else:
            raise Exception("dom_assign_act unknown: {}".format(dom_assign_act))

        self.cratio = cratio
        self.ndoms = ndoms
        self.lsqueeze = tf.keras.layers.GlobalAveragePooling2D()
        self.lconcat = tf.keras.layers.Concatenate(axis=0)
        self.lsigm = tf.keras.activations.sigmoid

    def build(self, input_shape):
        super(DomChCombAtt, self).build(input_shape)
        self.nfeats = input_shape[0][-1]
        if len(input_shape[0]) == 4:
            self.lreshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, self.nfeats]))
        elif len(input_shape[0]) == 5:
            self.lreshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, 1, self.nfeats]))
        else:
            raise Exception("function not implemented for this number of dimensions")
        self.nodes1 = self.nfeats // self.cratio
        self.adapters = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(self.nodes1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.nfeats)
            ], name='AdapterM' + str(nm))
            for nm in range(self.ndoms)]

        self.nmods = input_shape[1][-1]
        if self.cust_domassign == 0:
            self.dom_assign = tf.keras.Sequential([
                tf.keras.layers.Dense(self.ndoms),
                tf.keras.layers.Layer(self.dom_assign_act)
            ])
        elif self.cust_domassign == 1:
            self.nnodes1 = 2 ** self.nmods - 1
            self.dom_assign = tf.keras.Sequential([
                tf.keras.layers.Dense(self.nnodes1, activation='relu'),
                tf.keras.layers.Dense(self.ndoms),
                tf.keras.layers.Layer(lambda x: self.dom_assign_act(x))
            ])
        elif self.cust_domassign == 2:
            self.nnodes1 = 2 ** self.nmods - 1
            self.dom_assign = tf.keras.Sequential([
                tf.keras.layers.Dense(self.nnodes1, activation='relu'),
                tf.keras.layers.Dense(self.nnodes1, activation='relu'),
                tf.keras.layers.Dense(self.ndoms),
                tf.keras.layers.Layer(lambda x: self.dom_assign_act(x))
            ])

    @tf.function
    def call(self, inputs, **kwargs):
        x, x_chcomb = inputs
        xsq = self.lsqueeze(x)
        Xmods_arr = tf.TensorArray(
            size=self.ndoms,
            dynamic_size=False,
            clear_after_read=True,
            infer_shape=True,
            dtype=tf.float32,
            name='array_mods'
        )
        for nm, adaptmod in enumerate(self.adapters):
            Xmods_arr.write(nm, adaptmod(xsq))
        Xmods_aux = Xmods_arr.stack()
        Xmods_aux.set_shape([self.ndoms, Xmods_aux.shape[1], self.nfeats])
        xmods = tf.transpose(Xmods_aux, [1, 2, 0])

        # Assign block
        xassign = self.dom_assign(x_chcomb)
        xscale = tf.matmul(xmods, tf.expand_dims(xassign, 1), transpose_b=True)
        xscale = tf.gather(xscale, 0, axis=-1)
        xscale_sigm = self.lsigm(xscale)
        xscale_re = self.lreshape(xscale_sigm)
        xexc = tf.multiply(x, xscale_re)
        return xexc


def lchdrop(drop_type, rate=0.5):
    """
    Parses the type of Marker Sampling strategy employed

    Parameters
    ----------
    drop_type : str
        Sampling option (see code for options)
    rate : float
        Sampling rate

    Returns
    -------
    tf.keras.layers.Layer
        Marker Sampling layer to be included in the model
    """
    if drop_type == 'dropout':
        # drop_channels = tf.keras.layers.SpatialDropout2D(rate=rate, name="ChDrop")
        drop_channels = ChDrop_nonorm(rate=rate, do_train_scale=True, name="ChDrop_dropout")
    elif drop_type == 'nonorm':
        drop_channels = ChDrop_nonorm(rate=rate, name="ChDrop_nonorm")
    elif drop_type == 'renorm':
        drop_channels = ChDrop_nonorm(do_scale=True, rate=rate, name="ChDrop_renorm")
    elif drop_type == 'attnorm':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=2, activation='sigmoid', do_bias=False, merge_op='multiply')
        ], "ChDrop_att")
    elif drop_type == 'att_chcomb':
        drop_channels = ChDrop_nonorm(rate=rate, name="ChDrop_nonorm")
    elif drop_type == 'attnorm_64relubias':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=0, activation='relu', do_bias=True, merge_op='multiply')
        ], "ChDrop_att")
    elif drop_type == 'attnorm_64relubiasadd':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=0, activation='relu', do_bias=True, merge_op='add')
        ], "ChDrop_att")
    elif drop_type == 'attnorm_64':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=0, activation='sigmoid', do_bias=False, merge_op='multiply')
        ], "ChDrop_att")
    elif drop_type == 'attnorm_lrelu1':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=2, activation='sigmoid', activation1='lrelu', do_bias=False,
                       merge_op='multiply')
        ], "ChDrop_att")
    elif drop_type == 'attnorm_64lrelu2bias':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            SEchannels(cratio=0, activation='lrelu', activation1='lrelu', do_bias=True, merge_op='multiply')
        ], "ChDrop_att")
    elif drop_type == 'batchnorm':
        drop_channels = tf.keras.Sequential([
            ChDrop_nonorm(rate=rate, name="ChDrop_nonorm"),
            tf.keras.layers.BatchNormalization()
        ], "ChDrop_batchnorm")
    else:
        raise Exception("chdrop parameter {} unknown".format(drop_type))
    return drop_channels


def get_attention_layer(attname):
    """
    Parses the type of attention strategy employed

    Parameters
    ----------
    attname : str
        Attention block employed (see code for options)

    Returns
    -------
    tf.keras.layers.Layer
        Marker Sampling layer to be included in the model
    """
    if attname is None:
        fatt = None
    elif attname == 'ch':
        fatt = lambda x: SEchannels(name=x)
    elif attname == 'spat':
        fatt = lambda x: SEspatialMod(name=x)
    elif attname == 'dom':
        fatt = lambda x: SEdomain(name=x)
    elif attname == 'dom_sigm':
        fatt = lambda x: SEdomain(dom_assign_act='sigmoid', name=x)
    elif attname == 'att_chcomb':
        fatt = lambda x: ChCombAtt(name=x)
    elif attname == 'att_chcomb_postact':
        fatt = lambda x: ChCombAttPostAct(name=x)
    elif attname == 'att_domchcomb':
        fatt = lambda x: DomChCombAtt(name=x)
    elif attname == 'att_domchcomb_cust1':
        fatt = lambda x: DomChCombAtt(cust_domassign=1, name=x)
    elif attname == 'att_domchcomb_cust2':
        fatt = lambda x: DomChCombAtt(cust_domassign=2, name=x)
    elif attname == 'att_domchcomb_sm':
        fatt = lambda x: DomChCombAtt(dom_assign_act='softmax', name=x)
    elif attname == 'att_domchcomb_cust1_sm':
        fatt = lambda x: DomChCombAtt(cust_domassign=1, dom_assign_act='softmax', name=x)
    elif attname == 'att_domchcomb_cust1_relu1':
        fatt = lambda x: DomChCombAtt(cust_domassign=1, dom_assign_act='relu1', name=x)
    # elif attname == 'att_domchcomb_cust1_none':
    #     fatt = lambda x: DomChCombAtt(cust_domassign=1, dom_assign_act=None, name=x)
    else:
        raise Exception("Attention model unknown")
    return fatt


class ProjectionHead(tf.keras.layers.Layer):

    def __init__(self, nodes1=256, nodes2=128, down_factor=0, proj_type='dense', ndims=2, **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        # self.dense1 = tf.keras.layers.Dense(nodes1, activation='relu')
        # self.dense2 = tf.keras.layers.Dense(nodes2, activation=None)
        self.proj_type = proj_type
        if self.proj_type == 'dense':
            self.layer1 = tf.keras.layers.Dense(nodes1, activation='relu')
            self.layer2 = tf.keras.layers.Dense(nodes2, activation=None)
        elif self.proj_type == 'conv':
            if ndims == 3:
                self.layer1 = tf.keras.layers.Conv3D(nodes1, kernel_size=(1, 1, 1), activation='relu')
                self.layer2 = tf.keras.layers.Conv3D(nodes2, kernel_size=(1, 1, 1), activation=None)
            elif ndims == 2:
                self.layer1 = tf.keras.layers.Conv2D(nodes1, kernel_size=(1, 1), activation='relu')
                self.layer2 = tf.keras.layers.Conv2D(nodes2, kernel_size=(1, 1), activation=None)
            else:
                raise Exception("Method not implemented for {} dimensions".format(ndims))
        else:
            raise Exception("Method not implmeented for proj_type={}".format(proj_type))
        self.down_factor = down_factor
        if self.down_factor > 0:
            self.lmp = tf.keras.layers.MaxPool2D([down_factor, down_factor])
        self.flat_layer = tf.keras.layers.Flatten()

    @tf.function
    def call(self, inputs):
        if self.down_factor > 0:
            xaux = self.lmp(inputs)
        else:
            xaux = inputs
        if self.proj_type == 'dense':
            xf = self.flat_layer(xaux)
        else:
            xf = xaux
        xl = self.layer1(xf)
        return self.layer2(xl)
