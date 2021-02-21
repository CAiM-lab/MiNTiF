# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import numpy as np
import tensorflow as tf
from cnn import layers_keras as clayers, losses, metrics_keras as cmetrics
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomModel:
    """
    Base class to build other models on top
    """

    optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.keras.optimizers.Adam(0.000001)
    activation = 'linear'
    nlabels = 0
    loss_name = 'crossentropy'
    loss_weights = None
    att_minput = ('att_domchcomb', 'att_chcomb', 'att_domchcomb_cust1', 'att_domchcomb_sm', 'att_domchcomb_cust1sm',
                  'att_domchcomb_cust2')
    name_metrictrack = "Fscore_labelmean"

    def __init__(self, msettings, class_weights=None, name='CustomModel', **kwargs):
        """
        Class initialization

        Parameters
        ----------
        msettings : dict
            dictionary describing the model settings. Normally loaded from an external .json file
        class_weights : list or None
            list of floats indicating the weight for each of the classes
        name: str
            name assigned to identify the created model
        """

        # Data settings
        self.name = name
        self.msettings = msettings
        self.msettings.update({
            'activation': self.activation,
            'nchannels': len(msettings['channels']),
            'nlabels': len(msettings['labels']) + 1
        })
        self.class_weights = class_weights
        self.patch_size = msettings['patch_size']
        self.patch_size_out = msettings['patch_size_out']
        self.nchannels = len(msettings['channels'])
        self.input_aux = tf.keras.Input(shape=self.patch_size + [self.nchannels], name="input")
        self.input_shape_aux = self.input_aux.shape
        # self.input_shape = self.patch_size + [self.nchannels]

        # Network settings
        if 'nnNodes' in msettings:
            self.nn_nodes = np.array(msettings['nnNodes'])
        else:
            self.nn_nodes = np.array([[64, 128, 256, 512, 1024], [1024, 512, 256, 128, 64]])
        self.do_batch_norm = msettings['batchnorm']
        self.padding = msettings['padding']
        # Use extra label in segmentation (for bg)
        extra_labels = 0 if (
                ('dataset_type' in self.msettings) and self.msettings['dataset_type'] in ('detection')) else 1
        self.nlabels = len(msettings['labels']) + extra_labels
        self.scale_factor = msettings['scale_factor']
        self.ksize = msettings['kernel_size']
        self.nlevels = msettings['nlevels']

        # Graph settings
        self._graph_built = False
        self._model_built = False
        # self.run_eagerly = system.eager_mode
        # if self.run_eagerly:
        #     logger.warning("The model is running in eager mode, so it will be slower")
        if not (self.activation in ('softmax', 'sigmoid')):
            self.is_logits = True

    def get_loss(self):
        """
        Parses the string in self.loss_name to assign the proper loss

        Returns
        -------
        class:`tf.losses.Loss`
            loss function for the model
        """
        if self.loss_name == 'dice':
            return losses.DiceLoss(mode='macro_robust', only_foreground=True)
        elif self.loss_name == 'crossentropy':
            if self.class_weights is None:
                self.class_weights = [1] * self.nlabels
                logger.warning("class_weights should be declared for weighted cross entropy")
            return losses.CrossEntropyWeighted(self.class_weights)
        else:
            raise Exception("The loss {} has not been implemented".format(self.loss_name))

    def compute_output_shape(self, input_shape=None):
        # super(CustomModel, self).build(self.input_shape_aux)
        return self.patch_size_out + [self.nlabels]

    def get_summary_scalars(self, **kwargs):
        return None

    def call(self, inputs):
        logger.warning("Calling wrong method")
        return

    def model_fromlayer(self, nlayer, name=None):
        """
        Create a model from a specific layer

        Parameters
        ----------
        nlayer : int or str
            layer number or name that can be recognized within the model
        name str or None
            name assigned to the newly created model
        Returns
        -------
        class:`tf.keras.Model`
            newly created model
        """

        m_aux = self.build_model()
        if isinstance(nlayer, int):
            layer = m_aux.layers[nlayer]
        elif isinstance(nlayer, str):
            layer = m_aux.get_layer(nlayer)
        else:
            raise Exception("The layer type was not recognized")
        return tf.keras.Model(inputs=self.input_aux, outputs=layer.output, name=name)

    def build_model(self, name=None):
        """
        Created a keras model from the layers stablished in this class

        Parameters
        ----------
        name str or None
            name of the model

        Returns
        -------
        class:`tf.keras.Model`
            keras model
        """

        name = name or self.name
        model = tf.keras.Model(self.input_aux, self.call(self.input_aux), name=name)
        return model

    def build_metrics(self, do_idclass=False, **kwargs):
        """
        Builds the custom metrics required for the defined mdoel

        Parameters
        ----------
        do_idclass: bool
            Indicates if classes are treated separately (True) or aggregated together (False)

        Returns
        -------
        class:`tf.keras.metrics.Metric`
            Metrics for the model
        """
        return cmetrics.get_confmetrics(self.nlabels, self.is_logits, do_idclass=do_idclass)

    def set_metrics(self):
        return {
            "train": cmetrics.get_confmetrics(self.nlabels, self.is_logits),
            "val": cmetrics.get_confmetrics(self.nlabels, self.is_logits),
            "test": cmetrics.get_confmetrics(self.nlabels, self.is_logits)
        }

    def get_vislayers(self, model=None, names=None):
        """
        Defines the layers that will be visualized in tensorboard

        Parameters
        ----------
        model: class:`tf.keras.Model` or None
            model from which the visualizations are to be obtained
        names: list (of str) or None
            names of the layers to be visualized

        Returns
        -------
        class:`tf.keras.Model`
            Keras model that outputs the different defined layers
        """
        if names:
            layers = [layer.output for layer in model.layers
                      if not isinstance(layer.output, list) and
                      any([n in layer.name for n in names])]
        else:
            layers = [layer.output for layer in model.layers
                      if not isinstance(layer.output, list) and len(layer.output.shape) > 3]
        try:
            mlayers = tf.keras.Model(inputs=model.input, outputs=layers)
        except ValueError:
            logger.warning("Layers could not be obtained for visualization")
            mlayers = None
        return mlayers


class UNet(CustomModel):
    """
    Custom 2D U-Net model

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
    """

    def __init__(self,
                 nodes_factor=2,
                 do_outlayer=True,
                 channel_gather=None,
                 chdrop=None,
                 drop_rate=0.5,
                 do_att=None,
                 pos_att=None,
                 nn_nodes=None,
                 ndims=2,
                 unsupervised_settings=None,
                 **kwargs):
        """
        Class initialization

        Parameters
        ----------
        nodes_factor : int
            Reduce the number of nodes in every layer by the factor indicated here
        do_outlayer : bool
            Employ a last layer that outputs the classes (True) or output logits (False)
        channel_gather : int or None
            If int, the channel corresponding to that id will be employed by the network
        chdrop : str or None
            If str, defines the type of MarkerSampling employed as defined in `cnn.layers_keras.lchdrop`
        drop_rate : float
            If chdrop is True, this indicated the sampling ratio
        do_att : str or None
            If str, defines the type of attention employed as defined in `cnn.layers_keras.get_attention_layer`
        pos_att : list (of strs) or None
            It indicates the parts of the UNet where attention is employed. If None, it is used in ('encoder', 'bottleneck', 'decoder')
        nn_nodes : list (of lists (of ints)) or None
            Defines the number of nodes in every layer of the model.
        kwargs : key, value mappings
            Other keyword arguments are passed through to class:`CustomModel`
        """

        super().__init__(**kwargs)
        self.do_outlayer = do_outlayer
        if nn_nodes is not None:
            self.nn_nodes = nn_nodes
        else:
            self.nn_nodes = self.nn_nodes // nodes_factor

        if channel_gather is not None:
            if chdrop:
                raise Exception("chdrop and channel_gather cannot be used together")
            self.do_channelgather = True
            self.channelgather = clayers.ChannelGather(channel_gather)
            chname = "BE" + str(channel_gather) + "_"
        else:
            self.do_channelgather = False
            chname = ""

        self.chdrop = chdrop
        self.chdrop_multi = False
        self.drop_rate = drop_rate
        self.ndims = ndims
        if self.chdrop is not None:
            self.chdrop_multi = 'chcomb' in self.chdrop
            self.drop_channels = clayers.lchdrop(self.chdrop, drop_rate)
            if self.chdrop == 'att_chcomb':
                self.drop_channels_att = clayers.ChCombAtt(name="ChDrop_auxAtt")

        self.encoder = [None] * (self.nlevels)
        self.encoder_skip = [None] * (self.nlevels - 1)
        self.decoder = [None] * (self.nlevels - 1)
        self.crop_encoder = [None] * (self.nlevels - 1)
        self.concat = [tf.keras.layers.Concatenate() for _ in range(self.nlevels - 1)]
        if ndims == 2:
            self.down = [
                tf.keras.layers.MaxPool2D(self.scale_factor, padding=self.padding,
                                          name=chname + "downsample_level" + str(l))
                for l in range(self.nlevels - 1)]
        elif ndims == 3:
            self.down = [
                tf.keras.layers.MaxPool3D(self.scale_factor, padding=self.padding,
                                          name=chname + "downsample_level" + str(l))
                for l in range(self.nlevels - 1)]
        else:
            logger.error("Method not implemented for {} dimensions".format(ndims))
        self.up = [clayers.upsampling_custom(
            nodes=self.nn_nodes[1][l + 1],
            scale_factor=self.scale_factor,
            ndims=ndims,
            name=chname + "upsample_level" + str(l)
        ) for l in range(self.nlevels - 1)]

        self.do_att = do_att
        self.act2 = None if self.do_att == 'att_chcomb_postact' else 'relu'
        if self.do_att is None:
            self.pos_att = {k: False for k in ('encoder', 'bottleneck', 'decoder')}
            self.att_multi = False
        else:
            self.pos_att = pos_att or {k: True for k in ('encoder', 'bottleneck', 'decoder')}
            self.att_multi = 'chcomb' in self.do_att
            fatt = clayers.get_attention_layer(self.do_att)
            # self.latt = [[
            #     fatt("SE_" + net + str(nl)) for nl in range(self.nlevels)
            # ] for laux, net in zip((self.nlevels, self.nlevels - 1), ('e', 'd'))]
            self.latt_enc = [
                fatt("SE_enc_l" + str(nl)) for nl in range(self.nlevels - 1)
            ] if self.pos_att['encoder'] else None
            self.latt_dec = [
                fatt("SE_dec_l" + str(nl)) for nl in range(self.nlevels - 1)
            ] if self.pos_att['decoder'] else None
            self.latt_bottleneck = fatt("SE_bn") if self.pos_att['bottleneck'] else None

        self.encoder[0] = clayers.blockconv(self.nn_nodes[0][0],
                                            do_batchnorm=self.do_batch_norm,
                                            ksize=self.ksize,
                                            padding=self.padding,
                                            ndims=ndims,
                                            name=chname + "BlockDown_level0",
                                            rdrop=float(self.msettings["dropout"]))
        for level in range(1, self.nlevels):
            self.encoder[level] = clayers.blockconv(self.nn_nodes[0][level],
                                                    do_batchnorm=self.do_batch_norm,
                                                    ksize=self.ksize,
                                                    padding=self.padding,
                                                    activation2=self.act2,
                                                    ndims=ndims,
                                                    name=chname + "BlockDown_level" + str(level),
                                                    rdrop=float(self.msettings["dropout"]))
            self.decoder[level - 1] = clayers.blockconv(self.nn_nodes[1][level],
                                                        do_batchnorm=self.do_batch_norm,
                                                        ksize=self.ksize,
                                                        padding=self.padding,
                                                        activation2=self.act2,
                                                        ndims=ndims,
                                                        name=chname + "BlockUp_level" + str(level),
                                                        rdrop=float(self.msettings["dropout"]))

            self.crop_encoder[level - 1] = clayers.skipconnect(
                level - 1, self.nlevels, scale_factor=self.scale_factor, padding=self.padding, ndims=ndims,
                name=chname + "cropping_level" + str(level))
        if self.do_outlayer:
            if ndims == 2:
                self.final_layer = tf.keras.layers.Conv2D(filters=self.nlabels,
                                                          kernel_size=(1, 1),
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          name=chname + 'output_layer')
            elif ndims == 3:
                self.final_layer = tf.keras.layers.Conv3D(filters=self.nlabels,
                                                          kernel_size=(1, 1, 1),
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          name=chname + 'output_layer')
            else:
                logger.error("Method not implemented for {} dimensions".format(ndims))
        else:
            self.final_layer = tf.keras.layers.Lambda(lambda x: x, name=chname + 'output_layer')

    # @tf.function
    def call(self, inputs):
        x = inputs
        if self.chdrop:
            x = self.drop_channels(x)
        redaxis = [1, 2] if self.ndims == 2 else [1, 2, 3]
        x_chcomb = tf.cast(tf.not_equal(tf.reduce_sum(x, axis=redaxis), 0), tf.float32)
        if self.chdrop_multi:
            self.drop_channels_att([x, x_chcomb])
        # l_skips = tf.TensorArray(dtype=tf.float32, size=self.nlevels - 1, infer_shape=False)
        l_skips = [None] * (self.nlevels - 1)
        if self.do_channelgather:
            x = self.channelgather(x)
        x = self.encoder[0](x)
        for level, (fencoder, bdown) in enumerate(zip(self.encoder[1:], self.down)):
            if self.pos_att['encoder']:
                if self.att_multi:
                    xatt_aux = [x, x_chcomb]
                else:
                    xatt_aux = x
                x = self.latt_enc[level](xatt_aux)
            # l_skips = l_skips.write(level, x)
            l_skips[level] = x
            x = bdown(x)
            x = fencoder(x)
        if self.pos_att['bottleneck']:
            if self.att_multi:
                xatt_aux = [x, x_chcomb]
            else:
                xatt_aux = x
            x = self.latt_bottleneck(xatt_aux)

        max_level = self.nlevels - 1
        for level2, (fdecoder, bup, bconcat, skip_crop) in enumerate(zip(
                self.decoder, self.up, self.concat, reversed(self.crop_encoder))):
            max_level -= 1
            # x_skip = l_skips.read(max_level)
            x_skip = l_skips[max_level]
            x = bup(x)
            x_skip_crop = skip_crop(x_skip)
            # x_skip_crop = clayers.crop_match(x_skip, x)
            x = bconcat([x, x_skip_crop])
            x = fdecoder(x)
            if self.pos_att['decoder']:
                if self.att_multi:
                    xatt_aux = [x, x_chcomb]
                else:
                    xatt_aux = x
                x = self.latt_dec[level2](xatt_aux)
        xout = self.final_layer(x)
        return xout


class UNet2D(UNet):

    def __init__(self, ndims=2, **kwargs):
        super(UNet2D, self).__init__(ndims=ndims, **kwargs)

class UNet_MarkerDrop_MarkerExcite(UNet2D):
    """
    MS-ME model
    """

    def __init__(self, **kwargs):
        super(UNet_MarkerDrop_MarkerExcite, self).__init__(do_att="att_chcomb", chdrop="nonorm", **kwargs)


class MarkerSampling_MarkerExcite(UNet_MarkerDrop_MarkerExcite):
    """
    Alias for UNet_MarkerDrop_MarkerExcite
    """
    pass

class UNet3D(UNet):
    """
    Custom 3D U-Net model

    """

    def __init__(self, **kwargs):
        super(UNet3D, self).__init__(ndims=3, nodes_factor=4, **kwargs)
        self.name_metrictrack = 'fscore'

    def get_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def build_metrics(self, data_set=None, **kwargs):
        if data_set == 'train':
            return [tf.keras.metrics.MeanSquaredError()]
        else:
            return [cmetrics.CellMetrics()]