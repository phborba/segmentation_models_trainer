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
from tensorflow.python.util.tf_export import keras_export

__all__ = 'ImageHistory',

class ImageHistory(tf.keras.callbacks.Callback):

    def __init__(self, params, **kwargs):
        super(ImageHistory, self).__init__(**kwargs)
        self.tensor_board_dir = params['tensor_board_dir']
        self.data = params['data']
        self.last_epoch = 0 if 'current_epoch' not in params \
                            else params['current_epoch']
        self.n_epochs = params['n_epochs']
        self.draw_interval = params['draw_interval']
        self.batch_size = params['batch_size']
    
    def on_epoch_end(self, epoch, logs=None):
        self.my_logs = logs or {}
        self.last_epoch += 1
        predicted_images = []
        ref_labels = []
        image_data, label_data = list(self.data.take(1))[0]
        #took one batch
        y_pred = self.model.predict(image_data)
        predicted_images.append(y_pred)
        ref_labels.append(label_data)
        
        predicted_images = np.concatenate(predicted_images,axis=2)
        ref_labels = np.concatenate(ref_labels,axis=2)
        data = np.concatenate((predicted_images,ref_labels), axis=1)
        params = (image_data, label_data, y_pred)
        image_tensor = tf.py_function(
            self._wrap_pltfn(
                display_predictions_batch
            ),
            params,
            tf.uint8
        )
        image_tensor.set_shape([None, None, 4])
        file_writer = tf.summary.create_file_writer(self.tensor_board_dir)
        with file_writer.as_default():
            tf.summary.image(
                'epoch',
                data,
                step=self.last_epoch
            )
            tf.summary.image(
                'Ground Truth and Prediction Comparison',
                tf.expand_dims(image_tensor, 0),
                step=self.last_epoch
            )
            tf.summary.flush()
        image_tensor = None
        return
    
    def _wrap_pltfn(self, plt_fn):
        def plot(*args):
            fig = plt.figure(figsize=(20, 10*self.batch_size))
            gs = fig.add_gridspec(1, 3)
            fig, axs = plt.subplots(nrows=self.batch_size, ncols=3, figsize=(20, 100),
                    subplot_kw={'xticks': [], 'yticks': []})
            args = [fig, axs] + list(args)
            plt_fn(*args)
            buf = io.BytesIO()
            self.email_epoch_report(plt)
            plt.savefig(buf, format='png')
            buf.seek(0)
            im = tf.image.decode_png(buf.getvalue(), channels=4)
            buf.close()
            plt.close('all')
            fig = None
            axs = None
            return im
        return plot
    
    def email_epoch_report(self, plt):
        text_path = os.path.join(
                REPORT_DIR , 'metrics_report.txt'
        )
        report_path = os.path.join(
                REPORT_DIR , 'report_epoch_{epoch}_{date}.png'.format(
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=self.last_epoch
            )
        )
        email_title = 'Training of epoch {epoch}/{total_epochs} of experiment 9 has finished'.format(
            epoch=self.last_epoch,
            total_epochs=self.n_epochs
        )
        epoch_report =         """
        Metrics after epoch {epoch}:

        Loss: {loss}
        Accuracy: {acc}
        Binary Crossentropy: {bce}
        IoU: {iou}
        Precision: {precision}
        Recall: {recall}
        F1-Score: {f1_score}
        
        """.format(
            epoch=self.last_epoch,
            acc=self.my_logs.get('accuracy'),
            bce=self.my_logs.get('binary_crossentropy'),
            loss=self.my_logs.get('loss'),
            iou=self.my_logs.get('iou_score'),
            precision=self.my_logs.get('precision'),
            recall=self.my_logs.get('recall'),
            f1_score=self.my_logs.get('f1-score')
            
        )
        plt.savefig(
            report_path,
            format='png'
        )

        with open(text_path, "a") as myfile:
            myfile.write(epoch_report)
        # quickstart.send_epoch_report(email_title, epoch_report, report_path)
        # pool.submit(
        #     send_mail,
        #     (
        #         email_title,
        #         epoch_report,
        #         report_path
        #     )
        # )
        
        
    
def display_predictions_batch(plt, axs, *arg):
    sample_image, sample_mask, predicted_mask = arg
    for i in range(BATCH_SIZE):
        axs[i][0].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_image[i]
            )
        )
        axs[i][0].set_title("Image {n}".format(n=i+1))
        axs[i][1].imshow(
            tf.keras.preprocessing.image.array_to_img(
                sample_mask[i]
            )
        )
        axs[i][1].set_title("Ground Truth {n}".format(n=i+1))
        axs[i][2].imshow(
            tf.keras.preprocessing.image.array_to_img(
                predicted_mask[i]
            )
        )
        axs[i][2].set_title("Predicted {n}".format(n=i+1))
    plt.tight_layout()