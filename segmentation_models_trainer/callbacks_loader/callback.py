# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-14
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
from typing import Any, List
from segmentation_models_trainer.callbacks_loader.callback_factory import CallbackFactory
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Callback(JsonSchemaMixin):
    name: str
    config: dict

    @staticmethod
    def validate_callback_name(name):
        if name not in [
            i for i in dir(tf.keras.callbacks) + dir(tf.keras.callbacks.experimental)\
                if '__' not in i
            ]:
            raise ValueError("Callback not implemented")

    def get_callback(self):
        return CallbackFactory.get_callback(
            self.name,
            self.config
        )

@dataclass
class CallbackList(JsonSchemaMixin):
    items: List[Callback]

    def get_tf_objects(self):
        return [
            i.get_callback() for i in self.items
        ]



if __name__ == '__main__':
    import json
    x = Callback(
        name='ReduceLROnPlateau',
        config= {
            'monitor' : 'val_loss',
            'factor' : 0.2,
            'patience' : 5,
            'min_lr' : 0.001
        }
    )
    print(x.to_json())
    print(json.dumps([x.to_json()]))
    x.get_callback()
    y= [
        Callback.from_dict(
            {
                'name' : 'ReduceLROnPlateau',
                'config' : {
                    'monitor' : 'val_loss',
                    'factor' : 0.2,
                    'patience' : 5,
                    'min_lr' : 0.001
                }
            }
        ),
        Callback.from_dict(
            {
                'name' : 'ModelCheckpoint',
                'config' : {'filepath' : '/data/teste'}
            }
        )
    ]
    y
    z = CallbackList(y)
    print(z.to_json())
    w = Callback(
        'ImageHistory',
        config={
            'tensor_board_dir' : '/teste',
            'data' : None,
            'n_epochs' : 4,
            'draw_interval' : 1,
            'batch_size' : 1,
            'page_size' : 1,
            'report_dir': '/teste'
        }
    )
    w