# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-15
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

from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin
from typing import Any, List
from collections import OrderedDict
import tensorflow as tf
import importlib
import segmentation_models as sm

@dataclass
class SegmentationModel(JsonSchemaMixin):
    description: str
    backbone: str
    architecture: str
    activation: str = 'sigmoid'
    use_imagenet_weights: bool = True

    def __post_init__(self):
        if self.architecture not in ['Unet', 'PSPNet', 'FPN', 'Linknet', 'custom']:
            raise ValueError("Architecture not implemented")
        if self.backbone != 'custom' and \
            self.backbone not in sm.get_available_backbone_names():
            raise ValueError("Backbone not implemented")

    def get_model(self, n_classes, encoder_freeze, input_shape=None):
        """
        Gets keras model.
        Args:
            n_classes (int): number of classes
            encoder_freeze (bool): True if encoder weights are frozen,
            false otherwise.
            input_shape (shape tuple optional): Input shape. Defaults to None.
            If input_shape=None, (None, None, 3) shape is used.

        Raises:
            NotImplementedError: [description]

        Returns:
            [KerasModel]: Keras model implemented using either the 
            segmentation_model lib or custom architectures.
        """        
        input_shape = (None, None, 3) if input_shape is None else input_shape
        imported_model = getattr(
            sm, 
            self.architecture
        )
        return imported_model(
            self.backbone,
            input_shape=input_shape,
            encoder_weights='imagenet' if self.use_imagenet_weights else None,
            encoder_freeze=encoder_freeze
        )

if __name__ == '__main__':
    x = SegmentationModel.from_dict(
        {
            'backbone' : 'resnet18',
            'architecture' : 'Unet',
            'description' : 'test case'
        }
    )
    print(x.to_json())