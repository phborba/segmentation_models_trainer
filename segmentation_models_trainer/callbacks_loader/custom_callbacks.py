# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-16
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Philipe Borba - Cartographic Engineer 
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""
import tensorflow as tf
import numpy as np
import io
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.util.tf_export import keras_export
from itertools import zip_longest
from itertools import groupby, count

__all__ = 'ImageHistory',

def chunks(iterable, size):
    c = count()
    for _, g in groupby(iterable, lambda _: next(c)//size):
        yield g

class ImageHistory(tf.keras.callbacks.Callback):

    def __init__(self, params, **kwargs):
        super(ImageHistory, self).__init__(**kwargs)
        matplotlib.interactive(False)
        self.tensorboard_dir = params['tensorboard_dir'] if 'tensorboard_dir' in params else None
        self.dataset = params['dataset'] if 'dataset' in params else None
        self.n_epochs = params['n_epochs'] if 'n_epochs' in params else 1
        self.draw_interval = params['draw_interval'] if 'draw_interval' in params else 1
        self.batch_size = params['batch_size'] if 'batch_size' in params else 1
        self.page_size = params['page_size'] if 'page_size' in params else self.batch_size
        self.report_dir = params['report_dir'] if 'report_dir' in params else None
    
    def set_params(self, params):
        if 'dataset' in params:
            self.dataset = params['dataset']
        if 'tensorboard_dir' in params:
            self.tensorboard_dir = params['tensorboard_dir']
        if 'n_epochs' in params:
            self.n_epochs = params['n_epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'report_dir' in params:
            self.report_dir = params['report_dir']

    def on_epoch_end(self, epoch, logs=None):
        self.my_logs = logs or {}
        self.last_epoch = epoch

        image_data, label_data, y_pred, data = self.predict_data()

        file_writer = tf.summary.create_file_writer(
            self.tensorboard_dir
        )
        args = image_data, label_data, y_pred
        image_tensor = tf.py_function(
            self._wrap_pltfn(
                display_predictions
            ),
            args,
            tf.uint8
        )
        image_tensor.set_shape(
            [None, None, 4]
        )
        with file_writer.as_default():
            tf.summary.image(
                'epoch',
                data,
                step=self.last_epoch
            )
            tf.summary.image(
                'Ground Truth and Prediction Comparison for {n_images} images'.format(
                    n_images=self.n_pages
                ),
                tf.expand_dims(image_tensor, 0),
                step=self.last_epoch
            )
            tf.summary.flush()
        image_tensor = None
        return
    
    def predict_data(self):
        predicted_images = []
        ref_labels = []
        image_data, label_data = list(
            self.dataset.take(1)
        )[0]
        #took one batch
        y_pred = self.model.predict(image_data)
        for i in range(self.page_size):
            predicted_images.append(y_pred[i])
            ref_labels.append(label_data[i])
        
        # predicted_images = np.concatenate(
        #     predicted_images,
        #     axis=2
        # )
        # ref_labels = np.concatenate(
        #     ref_labels,
        #     axis=2
        # )

        data = np.concatenate(
            (
                predicted_images,
                ref_labels
            ),
            axis=1
        )
        
        return image_data, label_data, y_pred, data
    
    def _wrap_pltfn(self, plt_fn):
        def plot(*args):
            fig = plt.figure(figsize=(15, 15))
            fig, axs = plt.subplots(
                nrows=self.page_size,
                ncols=3,
                figsize=(20, 100),
                subplot_kw={'xticks': [], 'yticks': []}
            )
            arg_list = [fig, axs] + list(args)
            plt_fn(*arg_list)
            self.save_plot(plt)
            buf = io.BytesIO()
            plt.savefig(
                buf,
                format='png'
            )
            buf.seek(0)
            im = tf.image.decode_png(
                buf.getvalue(),
                channels=4
            )
            buf.close()
            plt.close('all')
            fig = None
            axs = None
            return im
        return plot
    
    def save_plot(self, plt):
        report_path = os.path.join(
            self.report_dir ,
            'report_epoch_{epoch}_{date}.png'.format(
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=self.last_epoch
            )
        )
        plt.savefig(
            report_path,
            format='png'
        )
    
def display_predictions(plt, axs, *arg):
    sample_image, sample_mask, predicted_mask  = arg
    for i in range(len(sample_image)):
        axs[i][0].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_image[i]
            )
        )
        axs[i][0].set_title(
            "Image {n}".format(
                n=i+1
            )
        )
        axs[i][1].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_mask[i]
            )
        )
        axs[i][1].set_title(
            "Ground Truth {n}".format(
                n=i+1
            )
        )
        axs[i][2].imshow(
            tf.keras.preprocessing.image.array_to_img(
                predicted_mask[i]
            )
        )
        axs[i][2].set_title(
            "Predicted {n}".format(
                n=i+1
            )
        )
    plt.tight_layout()