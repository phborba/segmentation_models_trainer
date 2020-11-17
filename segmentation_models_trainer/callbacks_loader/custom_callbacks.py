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
        self.n_pages = np.ceil(
            self.batch_size / self.page_size
        ) + 1
        for p, params in enumerate(
            zip_longest(
                chunks(image_data, int(self.n_pages)),
                chunks(label_data, int(self.n_pages)),
                chunks(y_pred, int(self.n_pages)),
                fillvalue=None
            )
        ):
            chunk_image_data, chunk_label_data, chunk_y_pred = params
            args = [
                list(chunk_image_data),
                list(chunk_label_data),
                list(chunk_y_pred),
                self.batch_size,
                p
            ]
            if any(i is None for i in args):
                break
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
                    'Ground Truth and Prediction Comparison Page {page}/{n_pages}'.format(
                        page=p,
                        n_pages=self.n_pages
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
        predicted_images.append(y_pred)
        ref_labels.append(label_data)
        
        predicted_images = np.concatenate(
            predicted_images,
            axis=2
        )
        ref_labels = np.concatenate(
            ref_labels,
            axis=2
        )

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
            *params, batch_size, current_page = args
            fig, axs = plt.subplots(
                nrows=batch_size.numpy(),
                ncols=3,
                figsize=(20, 100),
                subplot_kw={'xticks': [], 'yticks': []}
            )
            arg_list = [fig, axs, current_page] + params
            plt_fn(*arg_list)
            self.save_plot(plt, current_page, self.n_pages)
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
    
    def save_plot(self, plt, p, n_pages):
        report_path = os.path.join(
            self.report_dir ,
            'report_epoch_{epoch}_{page}-{n_pages}_{date}.png'.format(
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=self.last_epoch,
                page=p,
                n_pages=n_pages
            )
        )
        plt.savefig(
            report_path,
            format='png'
        )
    
def display_predictions(plt, axs, page_number, sample_image, sample_mask, predicted_mask):
    for i in range(len(sample_image)):
        axs[i][0].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_image[i]
            )
        )
        axs[i][0].set_title(
            "Image {n}".format(
                n=page_number+i+1
            )
        )
        axs[i][1].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_mask[i]
            )
        )
        axs[i][1].set_title(
            "Ground Truth {n}".format(
                n=page_number+i+1
            )
        )
        axs[i][2].imshow(
            tf.keras.preprocessing.image.array_to_img(
                predicted_mask[i]
            )
        )
        axs[i][2].set_title(
            "Predicted {n}".format(
                n=page_number+i+1
            )
        )
    plt.tight_layout()