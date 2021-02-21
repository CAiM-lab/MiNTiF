# (c) 2019-2021, Alvaro Gomariz  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

import os, logging, datetime, time, itertools
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import cnn_utils, common_utils
from cnn.keras_callbacks import basic_callback, LayersCallback
from config import system
from cnn import model_zoo

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelStructure:
    """
    High-level managing of model functions
    """

    def __init__(self, net, model_dir, linds=None, do_transfer=False, load_name='best', msettings=None, pretrained_model=None):
        """
        Class initialization

        Parameters
        ----------
        net : class:`cnn.model_zoo.CustomModel`
            CustomModel class defining the model to be employed
        model_dir : str
            Directory to save/load model related files
        linds : list (of lists (of ints))
            list defining the sample indices to be used as: [ids_train, ids_eval, ids_test]
        do_transfer : bool
            Whether transfer learning is applied
        """

        # Common
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.msettings = msettings
        self.model_dir = model_dir
        self.do_transfer = do_transfer
        if pretrained_model is None:
            self.model = net.build_model()
            self.model_vislayers = net.get_vislayers(self.model)
        else:
            self.model = tf.keras.models.load_model(pretrained_model)
            self.model_vislayers=None

        self.metrics = {
            'train': net.build_metrics(do_idclass=True, do_test=False, data_set='train'),
            'val': net.build_metrics(do_idclass=True, do_test=False, data_set='val')
        }
        self.metrics_test = net.build_metrics(do_idclass=True, do_test=True)
        self.is_logits = net.is_logits
        # Train
        self.loss = net.get_loss()
        # tmp
        self.optimizer = net.optimizer
        self.loss_weights = net.loss_weights
        self.linds = linds

        self.name_metrictrack = net.name_metrictrack
        if self.model:
            self.start_epoch, self.best_metric = self.load_net(load_name)
        else:
            self.start_epoch = 0
            self.best_metric = -np.Inf

    def compile_net(self, model=None):
        model = model or self.model
        self.logdir, self.writer = self.set_writer()
        self._tb = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=5)
        self._tb.set_model(model)
        self._tb.on_train_begin()

    def predict_write(self, x, datafile, l_meta, exp_name=None):
        """
        Predicts and writes to file

        Parameters
        ----------
        x : tf.Tensor or np.array
            Input image
        datafile : class:`data_read.datasets.dtfm.DTFM`
            Data class
        l_meta : list (of dicts)
            List where each entry corresponds to a dictionary with the metadata (sample and patch) for each image in the
            batch
        exp_name : str or None
            Experiment name
        """

        Y = self.model.predict(x)

        if datafile.msettings["dataset_type"] != "detection":
            # If the output of the model is logits, we apply softmax
            if self.is_logits:
                Y = tf.nn.softmax(Y).numpy() * 255
            # remove background
            Y=Y[...,:-1]
        else:
            Y-=Y.min()
            Y*=100/Y.max()

        # We write the result in the hdf5 file together with the required metadata
        for (y, wmeta) in zip(Y, l_meta):
            try:
                datafile.write_preds(y, wmeta, exp_name)
            except Exception as error:
                print("error in sample {}, patch {}".format(wmeta["sample"], wmeta["patch"]))
                raise Exception(error)

    def predict(self, ds_pred, datafile, batch_size, exp_name=None):
        """
        Manages the prediction for a tf.data.dataset

        Parameters
        ----------
        ds_pred : tf.data.dataset
            tf dataset with the data to be predicted. The second entry may be the metadata to point to the correct file
        datafile : class:`data_read.datasets.dtfm.DTFM`
            Data class
        batch_size : int
            Batch size
        exp_name : str or None
            Experiment name
        """

        bc = -1
        # Iterate for all patches in the dataset
        for x, metadata in ds_pred:
            # Create batches
            bc += 1
            if bc == 0:
                # For the first image in the batch we create an empty array
                l_meta = [None] * batch_size
                X = np.empty(shape=[batch_size] + list(x.shape), dtype=x.dtype)
            X[bc, ...] = x
            l_meta[bc] = metadata
            if bc == (batch_size - 1):
                # When the batch is complete we feed it to the model
                bc = -1
                # Predict + write
                self.predict_write(X, datafile, l_meta, exp_name=exp_name)
        if not bc == -1:
            # If the batch was not complete in the last data iteration, we feed the remaining images as the last batch
            self.predict_write(X, datafile, l_meta)

    def train(self, dataset, epochs, level='fit'):
        """
        Trains the model defined within the class with a given dataset

        Parameters
        ----------
        dataset : list (of tf.data.dataset)
            Datasets employed for training in a list as (train, validation, test)
        epochs : int
            Number of epochs to train
        level : str
            Different training functions (for developing purposes): 'fit' or 'tape'
        """

        if level in ('fit', 'model'):
            self.compile_model()
            self.train_modelfit(dataset, epochs)
        elif level in ('tape', 'low', 'net'):
            self.compile_net()
            self.train_net(dataset, epochs)
        else:
            raise Exception("Training level not found")

    def train_modelfit(self, dataset, epochs):
        """
        Train with keras.fit

        Parameters
        ----------
        dataset : list (of tf.data.dataset)
            Datasets employed for training in a list as (train, validation, test)
        epochs : int
            Number of epochs to train
        """

        ds_train, ds_val, ds_test = common_utils.input_dataset(dataset)
        callbacks = basic_callback(self.model_dir, best_metric='val_Fscore_labelmean', mode='max', best=-np.Inf)
        self.model.fit_generator(
            ds_train,
            epochs=self.start_epoch + epochs,
            steps_per_epoch=None,
            validation_data=ds_val,
            validation_steps=None,
            callbacks=callbacks + [LayersCallback(self.model_dir, ds_train.take(2), self.model_vislayers)],
            initial_epoch=self.start_epoch,
            verbose=system.verbose,
        )

    def load_net(self, mode='best'):
        """
        Load the model as defined in mode

        Parameters
        ----------
        mode : str
            Which model to load. "best" chooses the best model, and "last" takes the last epoch trained

        Returns
        -------
        int
            Starting epoch
        float
            Best metric so far (for validation)
        """

        if mode == 'best':
            fname = 'last_epoch_best.txt'
        elif mode == 'last':
            fname = 'last_epoch.txt'
        else:
            raise Exception("Mode unknown")
        flastepoch = os.path.join(self.model_dir, fname)
        last_model, start_epoch, best_val = cnn_utils.get_lastmodel(flastepoch, mode=mode)
        us_path = os.path.join(self.model_dir, '../unsupervised')
        if last_model is not None:
            logger.info(
                "Loading weights from {}, on epoch {} with a metric of {}".format(last_model, flastepoch, best_val))
            self.model.load_weights(last_model)
            if self.do_transfer:
                logger.info("Applying transfer learning, previous metrics will be deleted")
                best_val = -np.Inf
        elif os.path.isdir(us_path) and any(['model_query_last.tf' in x for x in os.listdir(us_path)]):
            logger.info("Transfer learning from unsupervised model: {}".format(
                os.path.join(us_path, 'model_query_last.tf')))
            model_name = getattr(model_zoo, self.msettings['unsupervised_model'])
            net_aux = model_name(msettings=self.msettings).model_query
            net_aux.load_weights(os.path.join(us_path, 'model_query_last.tf'))

            layer_dict = dict([(layer.name, layer) for layer in net_aux.layers])
            blockname = net_aux.layers[1].name
            layerblock_dict = dict([(layer.name, layer) for layer in net_aux.get_layer(blockname).layers])
            for layer in self.model.layers:
                layer_name = layer.name
                if layer_name in layer_dict:
                    logger.debug("Loading layer {} from main unsupervised model".format(layer_name))
                    layer.set_weights(layer_dict[layer_name].get_weights())
                elif layer_name in layerblock_dict:
                    logger.debug(
                        "Loading layer {} from block {} in main unsupervised model".format(layer_name, blockname))
                    layer.set_weights(layerblock_dict[layer_name].get_weights())

        best_metric = best_val or -np.Inf
        return start_epoch, best_metric

    def set_writer(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        logdir = os.path.join(
            self.model_dir, 'tensorboard', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        keys = ['train', 'val', 'graph', 'time', 'data_vis']
        writer = {k: tf.summary.create_file_writer(os.path.join(logdir, k)) for k in keys}
        return logdir, writer

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics_model,
            # run_eagerly=system.eager_mode
        )

    def test_func(self, ds_test, save_name=None, chdel=None, savevis_name=None, model=None):
        """
        Test the model with a given dataset

        Parameters
        ----------
        ds_test : tf.data.dataset
            Dataset with input and ground truth data to evaluate
        save_name : str or None
            Name employed to save the calculated metrics
        chdel : list (of ints) or None
            Defines markers to be deleted during evaluation
        savevis_name : str or None
            Name employed for saving visualizations
        """

        model = model or self.model
        save_name = save_name or 'metrics'
        time_test = time.time()
        if os.path.split(self.model_dir)[0][-2:] == 'l2':
            label_color = 'red'
        elif os.path.split(self.model_dir)[0][-2:] == 'l4':
            label_color = 'cyan'
        else:
            label_color = 'gray'
        if chdel is not None:
            lsh = ds_test._flat_shapes[0][1:-1]
            nchannels = model.input.shape[-1]
            mask = tf.expand_dims(tf.stack(
                [tf.cond(x in tuple(chdel), lambda: tf.zeros(lsh), lambda: tf.ones(lsh)) for x in range(nchannels)],
                axis=-1), axis=0)

        for cn, (x_batch_test, y_batch_test) in enumerate(ds_test):
            if chdel is not None:
                x_batch_test = x_batch_test * mask
            test_logits = model(x_batch_test, training=False)
            # Update test metrics
            [func_metric(y_batch_test, test_logits) for func_metric in self.metrics_test]


        # dmetrics_test = cnn_utils.display_metrics(self.metrics_test, 'test')
        dmetrics_test = {}
        for m in self.metrics_test:
            vres = m.result()
            if isinstance(vres, dict):
                dmetrics_test.update(vres)
            else:
                dmetrics_test[m.name] = vres
        pmetrics = pd.DataFrame.from_dict(dmetrics_test, orient='index', columns=['model']).T
        logger.debug("Time to test the model: {} seconds".format(time.time() - time_test))
        if chdel is not None:
            pmetrics['chdel'] = str(chdel)
        if self.linds is not None:
            pmetrics['inds_train'] = str(tuple(self.linds[0]))
            pmetrics['inds_val'] = str(tuple(self.linds[1]))
            pmetrics['inds_test'] = str(tuple(self.linds[2]))
        pmetrics.to_csv(os.path.join(self.model_dir, save_name + '.csv'))
        [x.reset_states() for x in self.metrics_test]

    def test_net(self, ds_test, do_chdel=False, save_name=None, savevis_name=None, chlist=None, model=None):
        """
        Manages how to test the model depending on the provided parameters

                ds_test : tf.data.dataset
            Dataset with input and ground truth data to evaluate
        save_name : str or None
            Name employed to save the calculated metrics
        chdel : list (of ints) or None
            Defines markers to be deleted during evaluation
        savevis_name : str or None
            Name employed for saving visualizations
                    ds_test : tf.data.dataset
            Dataset with input and ground truth data to evaluate
        save_name : str or None
            Name employed to save the calculated metrics
        chdel : list (of ints) or None
            Defines markers to be deleted during evaluation
        savevis_name : str or None
            Name employed for saving visualizations
        Parameters
        ----------
        ds_test : tf.data.dataset
            Dataset with input and ground truth data to evaluate
        do_chdel : bool
            Whether to apply marker deletion, which attempts evaluation on combinations of markers
        save_name : str or None
            Name employed to save the calculated metrics
        savevis_name : str or None
            Name employed for saving visualizations
        chlist : list (of ints) or None
            Which markers combinations to attempt. do_chdel must be True if provided
        """

        model = model or self.model
        if chlist and not do_chdel:
            logging.warning("A list was provided to chlist, but do_chdel was set to False. Setting to True...")
            do_chdel = True
        if not chlist:
            logger.info("Loading best weights for training: {}".format(self.model_dir, 'model_best.tf'))
            model.load_weights(os.path.join(self.model_dir, 'model_best.tf'))
            self.test_func(ds_test, save_name=save_name, chdel=None, savevis_name=savevis_name, model=model)
        if do_chdel:
            if not os.path.isdir(os.path.join(self.model_dir, 'chdel')):
                os.makedirs(os.path.join(self.model_dir, 'chdel'))
            nchannels = model.input.shape[-1]
            if chlist:
                chcombs = chlist
            else:
                # We create 2**n-1 combinations
                chcombs = list(set([tuple(set(x)) for x in itertools.product(np.arange(nchannels), repeat=nchannels)]))
            for ech, lch in enumerate(chcombs):
                if isinstance(lch, str):
                    # We consider the input to be the channel + 1
                    lch = [int(x) - 1 for x in lch if x.isdigit()]
                if savevis_name:
                    savedir_chcomb = os.path.join(savevis_name, 'chcomb_' + str(lch))
                    if not os.path.isdir(savedir_chcomb):
                        os.makedirs(savedir_chcomb)
                else:
                    savedir_chcomb = None
                logger.info("Test for channels {}. {} of {}".format(lch, ech, len(chcombs)))
                chdel = tuple(set(np.arange(nchannels)).difference(set(lch)))
                chsavename = os.path.join("chdel", 'metrics_ch' + "".join([str(x) for x in lch]))
                self.test_func(ds_test, save_name=chsavename, chdel=chdel,
                               savevis_name=savedir_chcomb, model=model)

    def vis_net(self, ds_test, save_name):
        """
        Visualize results of the model with matplotlib

        Parameters
        ----------
        ds_test : tf.data.dataset
            Dataset to visualize
        save_name : str
            Name employed for the visualization files
        """

        self.model.load_weights(os.path.join(self.model_dir, 'model_best.tf'))
        for cn, (x_batch_test, fmeta) in enumerate(ds_test):
            if (len(self.model.input_shape) - len(x_batch_test.shape)) == 1:
                x_batch_test = np.expand_dims(x_batch_test, 0)
            test_logits = self.model(x_batch_test, training=False)

    def train_net(self, dataset, epochs):
        """
        Train model with custom pipeline

        Parameters
        ----------
        dataset : list (of tf.data.datasets)
            Datasets provided as a list: (train, validation, test)
        epochs : int
            How many epochs to train the model
        """
        tstart = time.time()
        self._count_batch = -1
        ds_train, ds_val, ds_test = common_utils.input_dataset(dataset)
        cb_layers = LayersCallback(self.model_dir, ds_train.take(1),
                                   self.model_vislayers) if self.model_vislayers is not None else None
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            tepoch = time.time()
            self._tb.on_epoch_begin(epoch)
            logger.info("\nStart of epoch {:d}".format(epoch))
            # Train
            ttrain = time.time()
            self.train_epoch(ds_train, epoch)
            ## End of epoch
            # Write time
            self.tb_time(time.time() - ttrain, 'timetrain_epoch', epoch)
            # Validation
            tval = time.time()
            self.eval_step(ds_val, 'val', step=epoch)
            self.tb_time(time.time() - tval, 'timeval_epoch', epoch)
            # Write metrics
            [self.write_metrics(k, epoch) for k in ['train', 'val']]
            # Save model
            self.track_save(epoch, self.name_metrictrack)
            # Reset metrics
            [func_metric.reset_states() for func_metric in self.metrics['train'] + self.metrics['val']]
            # Visualize layers
            if cb_layers is not None:
                cb_layers.on_epoch_end(epoch)
            self._tb.on_epoch_end(epoch)
            # Flush writer
            [w.flush() for w in self.writer.values()]
            self.tb_time(time.time() - tepoch, 'timetotal_epoch', epoch)
        self._tb.on_train_end()
        tend = time.time()
        logger.info("Time for training: {}".format(tend - tstart))
        # Test
        self.model.load_weights(os.path.join(self.model_dir, 'model_best.tf'))
        self.eval_step(ds_test, 'test')
        # Return test metrics in pandas format
        try:
            cnn_utils.dataset_summary(dataset, self.logdir, model=self.model)
        except:
            logger.warning("The summary of the dataset could not be created. This is a critical error and continue "
                           "only to save the test metrics")

    def track_save(self, epoch, kmetric='Fscore_labelmean', model=None, dmetrics=None):
        """
        Tracks when to save the model

        Parameters
        ----------
        epoch : int
            Current epoch
        kmetric : str or None
            Name of the tracked metric. If None, 'Fscore_labelmean' is taken
        """
        model = model or self.model
        dmetrics = dmetrics or self.metrics
        # Best
        track_metric = None
        for m in dmetrics['val']:
            mval = m.result()
            if isinstance(mval, dict):
                if kmetric in mval:
                    track_metric = mval[kmetric]
            elif m.name == kmetric:
                track_metric = mval
        if track_metric is None:
            logger.error('Metric {} was not found'.format(kmetric))
        if track_metric > self.best_metric:
            logger.info("Best metric improved from {} to {}".format(self.best_metric, track_metric))
            self.best_metric = track_metric
            model.save_weights(os.path.join(self.model_dir, 'model_best.tf'))
            with open(os.path.join(self.model_dir, 'last_epoch_best.txt'), 'w') as f:
                f.write('epoch {} -- value {}'.format(epoch, self.best_metric))

        # Frequency
        # if not (epoch % 10):
        #     self.model.save_weights(os.path.join(self.model_dir, 'model.tf'))
        #     with open(os.path.join(self.model_dir, 'last_epoch.txt'), 'w') as f:
        #         f.write(str(epoch))

    @tf.function
    def train_step(self, x, y):
        """
        Train step of the model

        Parameters
        ----------
        x : tf.Tensor or np.array
            Input data
        y : tf.Tensor or np.array
            Ground truth

        Returns
        -------
        tf.Tensor
            Output of the model
        float
            Loss for this train step
        """

        # assert tf.reduce_all(tf.equal(tf.reduce_sum(y, axis=-1), tf.ones(y.shape[:-1])))
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # print(loss_value)
        # print(max([np.abs(x.numpy()).max() for x in grads]))
        return logits, loss_value

    def train_epoch(self, ds_train, epoch=None):
        """
        Complete training epoch

        Parameters
        ----------
        ds_train : tf.data.Dataset
            Input and output data provided as a tf dataset
        epoch : int or None
            Current epoch, for logging purposes
        """

        tstart = time.time()
        total_loss = 0.
        step = -1
        for step, data_batch in enumerate(ds_train):
            x_batch_train, y_batch_train = cnn_utils.split_batch(data_batch)
            self._tb.on_batch_begin(self._count_batch)
            self._count_batch += 1
            logits, loss_value = self.train_step(x_batch_train, y_batch_train)
            total_loss += loss_value
            # Update training metrics
            # def vis_im(im):
            #     imcut = im[:, im.shape[1]//2,...]
            #     imcut_aux = imcut - tf.reduce_min(imcut)
            #     return imcut_aux / tf.reduce_max(imcut_aux)
            # with self.writer['train'].as_default():
            #     tf.summary.image("trainvis_x", vis_im(x_batch_train), step=step)
            #     tf.summary.image("trainvis_ygt", vis_im(y_batch_train), step=step)
            #     tf.summary.image("trainvis_ypred", vis_im(logits), step=step)

            #
            if self.metrics is not None and "train" in self.metrics:
                [func_metric(y_batch_train, logits) for func_metric in self.metrics["train"]]
            with self.writer['train'].as_default():
                tf.summary.scalar('loss_batch', loss_value, self._count_batch)
            time_batch = time.time() - tstart
            self.tb_time(time_batch, 'time_batch', self._count_batch)
            self._tb.on_batch_end(self._count_batch)
            tstart = time.time()
        total_loss /= (step + 1)
        if epoch is not None:
            with self.writer['train'].as_default():
                tf.summary.scalar('loss', total_loss, epoch)

    def eval_step(self, ds, kmetric, step=None, dmetrics=None, model=None):
        """
        Evaluation of the model, which is saved in the class metrics and tensorboard

        Parameters
        ----------
        ds : tf.data.Dataset
            Dataset to be evaluated
        kmetric : str
            Group of metrics to be evaluated (e.g. 'val' or 'test')
        step : int
            Current epoch, for logging purposes
        do_loss : bool
            Define whether the loss is calculated in this evaluation step, as some evaluation datasets don't have the
            same gt
        """
        model = model or self.model
        dmetrics = dmetrics or self.metrics
        if dmetrics is not None and kmetric in dmetrics:
            loss_value = 0.
            count = 0
            for data_batch in ds:
                x_batch, y_batch = cnn_utils.split_batch(data_batch)
                # Lazy definition of having a different gt for some datasets
                do_loss = y_batch.ndim == x_batch.ndim
                count += 1
                val_logits = model(x_batch, training=False)
                if do_loss:
                    loss_value += self.loss(y_batch, val_logits)
                # Update val metrics
                # self.metrics['val'][0].test(y_batch, val_logits)
                [func_metric(y_batch, val_logits) for func_metric in dmetrics[kmetric]]
            if do_loss:
                loss_value /= count
            if do_loss and step is not None:
                with self.writer['val'].as_default():
                    tf.summary.scalar('loss', loss_value, step)

    def write_metrics(self, kmetric, epoch, dmetrics=None):
        """
        Write metrics in tensorboard

        Parameters
        ----------
        kmetric : str
            Group of metrics (e.g. 'val' or 'test')
        epoch : int
            Current epoch, for logging purposes
        """

        printmetrics = ""
        dmetrics = dmetrics or self.metrics
        for metric in dmetrics[kmetric]:
            if isinstance(metric.result(), dict):
                pass  # Not implemented
            else:
                if 'Fscore' in metric.name:
                    print_aux = "label {}: {:0.3f} __ ".format(metric.id_class, metric.result())
                    if metric.layer > -1:
                        print_aux = "layer {}, ".format(metric.layer) + print_aux
                    printmetrics += print_aux
                    logger.info("{} Fscore: {}".format(kmetric, printmetrics))

                #todo does this make sense?
                else:
                    try:
                        print_aux = "label {}: {:0.3f} __ ".format(metric.name, metric.result())
                        # if metric.layer > -1:
                        #     print_aux = "layer {}, ".format(metric.layer) + print_aux
                        printmetrics += print_aux
                        logger.info("{} {}: {}".format(kmetric, metric.name, printmetrics))
                    except:
                        logger.info("could not parse metric")



        # Tensorboard
        with self.writer[kmetric].as_default():
            for m in dmetrics[kmetric]:
                vres = m.result()
                if isinstance(vres, dict):
                    for k, v in vres.items():
                        tf.summary.scalar(k, v, step=epoch)
                else:
                    tf.summary.scalar(m.name, m.result(), step=epoch)
        self.writer[kmetric].flush()

    def tb_time(self, mtime, name, step):
        """
        Writes time in tensorboard

        Parameters
        ----------
        mtime : float
            Time elapsed
        name : str
            Name of field in tensorboard
        step : str
            Step, for logging purposes
        """

        if self.writer is not None and 'time' in self.writer:
            with self.writer['time'].as_default():
                tf.summary.scalar(name, mtime, step)
