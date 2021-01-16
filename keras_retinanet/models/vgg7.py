"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from tensorflow import keras

from . import saffronnet
from . import Backbone
from ..utils.image import preprocess_image


class VGG7Backbone(Backbone):
    """ Describes backbone information and provides utility functions.
        VGG7 is the first seven layers of VGG16 with MaxPooling2D at the end :)
        Note: MaxPooling layers are not counted :/
    """
    def __init__(self, backbone):
        super().__init__(backbone)
        print('Use VGG7 as backbone')


    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg7':
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return keras.utils.get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg7']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def vgg_retinanet(num_classes, backbone='vgg7', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg7, vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    output_layer_name = "backbone_output"
    if backbone == 'vgg7':
        vgg16 = keras.applications.vgg16(input_tensor=inputs, include_top=False, weights=None)
        outputs = keras.layers.MaxPool2D(name=output_layer_name)(vgg16.layers[8].output)
        vgg = keras.Model(inputs=vgg16.inputs, outputs=outputs)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = [output_layer_name]
    layer_outputs = [vgg.get_layer(name).output for name in layer_names]

    # C2, C3, C4, C5 not provided
    backbone_output = layer_outputs[0]
    
    full_model = saffronnet.saffronnet(inputs=inputs, num_classes=num_classes, backbone_output=backbone_output, **kwargs) 
    return full_model
