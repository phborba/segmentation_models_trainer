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
from typing import Any, List
from dataclasses import dataclass
from dataclasses_jsonschema import JsonSchemaMixin

@dataclass
class Metric(JsonSchemaMixin):
    class_name: str
    config: dict
    framework: str
    
    def __post_init__(self):
        if self.framework not in ['sm', 'tf.keras']:
            raise ValueError("Metric not implemented")

    def get_metric(self):
        if self.framework == 'sm':
            return self.get_sm_metric(self.class_name)
        elif self.framework == 'tf.keras':
            identifier = {
                "class_name" : self.class_name,
                "config" : self.config
            }
            return tf.keras.metrics.get(identifier)
        else:
            raise ValueError("Metric not implemented")

    def get_sm_metric(self, name):
        if self.class_name == 'iou_score':
            return sm.metrics.iou_score
        elif self.class_name == 'precision':
            return sm.metrics.precision
        elif self.class_name == 'recall':
            return sm.metrics.recall
        elif self.class_name == 'f1_score':
            return sm.metrics.f1_score
        elif self.class_name == 'f2_score':
            return sm.metrics.f2_score
        else:
            raise ValueError("SM metric not implemented")

@dataclass
class MetricList(JsonSchemaMixin):
    items: List[Metric]

    def get_tf_objects(self):
        return [
            i.get_metric() for i in self.items
        ]

if __name__ == "__main__":
    import json
    metric_list = [
        Metric(
            class_name=i,
            config={},
            framework='sm'
        ) for i in ['iou_score', 'precision', 'recall', 'f1_score', 'f2_score']
    ] + [
         Metric(
            class_name=i,
            config={} if i != 'MeanIoU' else {'num_classes':2},
            framework='tf.keras'
        ) for i in ['LogCoshError', 'KLDivergence', 'MeanIoU']
    ]
    x=MetricList(metric_list)
    print(x.to_json())