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
import segmentation_models_trainer as smt
from segmentation_models_trainer.callbacks_loader.custom_callbacks import ImageHistory 

class CallbackFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_callback(name, parameters):
        available_callbacks = [
            'ReduceLROnPlateau',
            'ModelCheckpoint',
            'TensorBoard'
        ] + [
            i for i in smt.callbacks_loader.custom_callbacks.__all__
        ]
        if name not in available_callbacks:
            raise ValueError("Callback not implemented")
        if name == 'ImageHistory':
            return smt.callbacks_loader.custom_callbacks.ImageHistory(
                parameters
            )
        if name == 'ReduceLROnPlateau':
            return tf.keras.callbacks.ReduceLROnPlateau(
                **parameters
            )
        if name == 'ModelCheckpoint':
            return tf.keras.callbacks.ModelCheckpoint(
                **parameters
            )
        if name == 'TensorBoard':
            return tf.keras.callbacks.TensorBoard(
                **parameters
            )

if __name__ == '__main__':
    img_callback = CallbackFactory.get_callback(
        'ImageHistory',
        parameters={
            'tensor_board_dir' : '/teste',
            'data' : None,
            'n_epochs' : 4,
            'draw_interval' : 1,
            'batch_size' : 1,
            'page_size' : 1,
            'report_dir': '/teste'
        }
    )