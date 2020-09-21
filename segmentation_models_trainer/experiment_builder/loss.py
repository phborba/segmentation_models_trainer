# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-21
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Philipe Borba - Cartographic Engineer @ Brazilian Army
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
import segmentation_models as sm
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Loss(JsonSchemaMixin):
    class_name: str
    config: dict
    framework: str

    def __post_init__(self):
        if self.framework == 'sm':
            self.loss_obj = self.get_sm_loss(self.class_name)
        elif self.framework == 'tf.keras':
            identifier = {
                "class_name" : self.class_name,
                "config" : self.config
            }
            self.loss_obj = tf.keras.losses.get(identifier)
        else:
            raise ValueError("Loss not implemented")
        

    def get_sm_loss(self, name):
        if self.class_name == 'jaccard_loss':
            return sm.losses.JaccardLoss(**self.config)
        elif self.class_name == 'dice_loss':
            return sm.losses.DiceLoss(**self.config)
        elif self.class_name == 'binary_focal_loss':
            return sm.losses.BinaryFocalLoss(**self.config)
        elif self.class_name == 'categorical_focal_loss':
            return sm.losses.CategoricalFocalLoss(**self.config)
        elif self.class_name == 'binary_crossentropy':
            return sm.losses.BinaryCELoss(**self.config)
        elif self.class_name == 'categorical_crossentropy':
            return sm.losses.CategoricalCELoss(**self.config)
        elif self.class_name == 'bce_dice_loss':
            return sm.losses.BinaryCELoss(**self.config) + sm.losses.DiceLoss(**self.config)
        elif self.class_name == 'bce_jaccard_loss':
            return sm.losses.BinaryCELoss(**self.config) + sm.losses.JaccardLoss(**self.config)
        elif self.class_name == 'cce_dice_loss':
            return sm.losses.CategoricalCELoss(**self.config) + sm.losses.DiceLoss(**self.config)
        elif self.class_name == 'cce_jaccard_loss':
            return sm.losses.CategoricalCELoss(**self.config) + sm.losses.JaccardLoss(**self.config)
        elif self.class_name == 'binary_focal_dice_loss':
            return sm.losses.BinaryFocalLoss(**self.config) + sm.losses.DiceLoss(**self.config)
        elif self.class_name == 'binary_focal_jaccard_loss':
            return sm.losses.BinaryFocalLoss(**self.config) + sm.losses.JaccardLoss(**self.config)
        elif self.class_name == 'categorical_focal_dice_loss':
            return sm.losses.CategoricalFocalLoss(**self.config) + sm.losses.DiceLoss(**self.config)
        elif self.class_name == 'categorical_focal_jaccard_loss':
            return sm.losses.CategoricalFocalLoss(**self.config) + sm.losses.JaccardLoss(**self.config)
        else:
            raise ValueError("SM Loss not implemented")

if __name__ == "__main__":
    import json
    x = Loss(
        class_name='bce_dice_loss',
        config={},
        framework='sm'
    )
    print(x.to_json())
    x