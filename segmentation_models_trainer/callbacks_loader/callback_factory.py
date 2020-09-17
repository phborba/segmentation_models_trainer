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
from segmentation_models_trainer.callbacks_loader.custom_callbacks import *  

class CallbackFactory:
    def __init__(self):
        self.available_callbacks = [
                i for i in dir(tf.keras.callbacks) if '__' not in i and i != 'experimental'
            ] + [
                i for i in dir(tf.keras.callbacks.experimental) if not i.startswith('_')
            ] + self.get_custom_callbacks()
    
    @staticmethod
    def get_custom_callbacks():
        return [
            i for i in smt.callbacks_loader.custom_callbacks.__all__
        ]

    @staticmethod
    def get_callback(name, parameters):
        if name not in self.available_callbacks:
            raise ValueError("Callback not implemented")
        if name == 'ImageHistory':
            return segmentation_models_trainer.callbacks_loader.custom_callbacks.ImageHistory(
                
            )

if __name__ == '__main__':
    x=CallbackFactory()