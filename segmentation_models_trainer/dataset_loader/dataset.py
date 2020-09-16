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

AUGMENTATIONS = OrderedDict(
    {
        'random_crop' : lambda x: tf.image.random_crop(
            x[0],
            size=[
                x[1]['crop_width'],
                x[1]['crop_height'],
                4
            ]
        ),
        'random_flip_left_right' : lambda x: tf.image.random_flip_left_right(x[0]),
        'random_flip_up_down' : lambda x: tf.image.random_flip_up_down(x[0]),
        'random_brightness' : lambda x: tf.image.random_brightness(
            x[0],
            max_delta=x[1]['max_delta']
        ),
        'random_contrast' : lambda x: tf.image.random_contrast(
            x[0],
            lower=x[1]['lower'],
            upper=x[1]['upper']
        ),
        'random_saturation' : lambda x: tf.image.random_saturation(
            x[0],
            lower=x[1]['lower'],
            upper=x[1]['upper']
        ),
        'random_hue' : lambda x: tf.image.random_hue(
            x[0],
            max_delta=x[1]['max_delta']
        ),
        'per_image_standardization' : lambda x: tf.image.per_image_standardization(
            x[0]
        )
    }
)

IMAGE_DTYPE = {
    'float32' : tf.float32,
    'float16' : tf.float16
}

@dataclass
class ImageAugumentation(JsonSchemaMixin):
    name: str
    parameters: dict

    def augment_image(self, input_image, input_mask):
        image = tf.concat(
                values=[input_image, input_mask], axis=-1
            ) if self.name in [
                'random_crop',
                'random_flip_left_right',
                'random_flip_up_down'
            ] else input_image
        augmented_image = AUGMENTATIONS[self.name](
            [image, self.parameters]
        )
        if self.name in [
                'random_crop', 'random_flip_left_right', 'random_flip_up_down'
            ]:
            image = augmented_image[:, :, :-1]
            mask = augmented_image[:, :, -1:]
        else:
            image = augmented_image
            mask = input_mask
        return image, mask

@dataclass
class Dataset(JsonSchemaMixin):
    name: str
    file_path: str
    n_classes: int
    augmentation_list: List[ImageAugumentation]
    cache: Any = True
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    shuffle_csv: bool = True
    ignore_errors: bool = True
    num_paralel_reads: int = 4
    img_dtype: str = 'float32'
    img_format: str = 'png'
    img_width: int = 256
    img_length: int = 256
    use_ds_width_len: bool = False
    autotune: int = tf.data.experimental.AUTOTUNE
    distributed_training: bool = False

    def get_tf_dataset(self, batch_size):
        ds = tf.data.experimental.make_csv_dataset(
            self.file_path,
            batch_size=batch_size,
            ignore_errors=self.ignore_errors,
            num_parallel_reads=self.num_paralel_reads
        )
        labeled_ds = ds.map(
            self.process_csv_entry,
            num_parallel_calls=self.autotune
        )
        prepared_ds = self.prepare_for_training(
            labeled_ds,
            batch_size
        )
        if not self.distributed_training:
            return prepared_ds
        else:
            strategy = tf.distribute.get_strategy()
            return strategy.experimental_distribute_dataset(prepared_ds)

    def process_csv_entry(self, entry):
        width = entry['width'] if self.use_ds_width_len else self.img_width
        length = entry['length'] if self.use_ds_width_len else self.img_length
        label = tf.io.read_file(
            entry['label_path'][0]
        )
        label = self.decode_img(label, width, length, channels=1)
        # load the raw data from the file as a string
        img = tf.io.read_file(
            entry['image_path'][0]
        )
        img = self.decode_img(img, width, length)
        img, label = self.augment_image(img, label)
        return img, label
    
    @tf.function
    def decode_img(self, img, width, length, channels=3):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, IMAGE_DTYPE[self.img_dtype])
        # resize the image to the desired size.
        return tf.image.resize(img, [width, length])
    
    def prepare_for_training(self, ds, batch_size):
        if self.cache:
            if isinstance(self.cache, str):
                ds = ds.cache(self.cache)
            else:
                ds = ds.cache()
        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer_size)
        ds = ds.batch(batch_size)
        # Repeat forever
        ds = ds.repeat()
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds
    
    def augment_image(self, img, label):
        for augmentation in self.augmentation_list:
            img, label = augmentation.augment_image(
                img,
                label
            )
        return img, label

if __name__ == '__main__':
    aug_list = [
        ImageAugumentation.from_dict(
            {
                'name' : 'random_crop',
                'parameters' : {
                    'crop_width' : 256,
                    'crop_height' : 256
                }
            }
        ),
        ImageAugumentation.from_dict(
            {
                'name' : 'per_image_standardization',
                'parameters' : {
                }
            }
        )
    ]
    y = Dataset(
        name='test',
        file_path='/data/test',
        n_classes=1,
        augmentation_list=aug_list
    )
    print(y.to_json())