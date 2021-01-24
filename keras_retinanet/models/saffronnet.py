"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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
from .. import initializers
from .. import layers
from ..utils.center_alpha_anchors import AnchorParameters
from . import assert_training_model


def default_classification_model(
    num_classes,
    num_anchors,
    backbone_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification'
):
    """ Creates the default classification submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        backbone_feature_size        : The number of filters to expect from the feature backbone levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(backbone_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, backbone_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='classification_{}'.format(i),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='classification_sigmoid')(outputs)
    
    print("Classification submodel is placed in backbone")
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(
    num_values,
    num_anchors,
    backbone_feature_size=256,
    regression_feature_size=256,
    name='regression'
    ):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        backbone_feature_size    : The number of filters to expect from the feature backbone levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(backbone_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, backbone_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='regression_orginal', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='regression_reshape')(outputs)
    
    print("Regression submodel is placed in backbone")
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)




def default_submodels(num_values, num_classes, num_anchors, backbone_feature_size):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return {
        'regression': default_regression_model(num_values, num_anchors, backbone_feature_size),
        'classification': default_classification_model(num_classes, num_anchors, backbone_feature_size)
    }

def __build_anchors(anchor_parameters, on_layer):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        on_layer: keras(model) layer that you ant build anchors on it(build anchors according to this layer shape)
    Returns
        A tensor containing the anchors for the FPN features.
        The shape is:
        ```
        (batch_size, num_anchors, 3)
        ```
    """
    anchors = layers.Anchors(
                stride=anchor_parameters.strides[0],
                alpha_segments=anchor_parameters.alpha_segments,
                name='anchors'
            )(on_layer)
    return anchors



def saffronnet(
    inputs,
    backbone_output,
    num_classes,
    num_anchors             = None,
    submodels               = None,
    name                    = 'saffronnet',
    num_values              = 3
):
    """ Construct a SaffronNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        # Depricated # create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5, and possibly C2 from the backbone.
        # Depricated # pyramid_levels          : pyramid levels to use.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.
        num_values              : Number of regressoin parameters
    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    if keras.backend.image_data_format() == 'channels_first':
        backbone_feature_size = backbone_output.shape[1]
    else:
        backbone_feature_size = backbone_output.shape[-1]
    

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(
            num_values              = num_values,
            num_classes             = num_classes,
            num_anchors             = num_anchors,
            backbone_feature_size   = backbone_feature_size)
        
        regression = submodels['regression'](backbone_output)
        classification = submodels['classification'](backbone_output) 

    return keras.models.Model(inputs=inputs, outputs=[regression, classification], name=name)


def saffronnet_center_alpha(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'saffronnet-center-alpha',
    anchor_params         = None,
    nms_threshold         = 0.5,
    score_threshold       = 0.05,
    max_detections        = 300,
    parallel_iterations   = 32,
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        pyramid_levels        : pyramid levels to use.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        parallel_iterations   : Number of batch items to process in parallel.
        **kwargs              : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = saffronnet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # last layer of regression submodel before Reshape
    regression_orginal = model.get_layer('regression').get_layer('regression_orginal')
    
    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    anchors  = __build_anchors(anchor_params, on_layer=regression_orginal.output)

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    lines = layers.RegressLines(name='lines')([anchors, regression, classification])
    # lines = layers.ClipLines(name='clipped_lines')([model.inputs[0], lines])

    # # filter detections (apply NMS / score threshold / select top-k)
    # detections = layers.FilterDetections(
    #     nms                   = nms,
    #     class_specific_filter = class_specific_filter,
    #     name                  = 'filtered_detections',
    #     nms_threshold         = nms_threshold,
    #     score_threshold       = score_threshold,
    #     max_detections        = max_detections,
    #     parallel_iterations   = parallel_iterations
    # )([boxes, classification] + other)

    # construct the model

    keras.utils.plot_model(model, to_file='model.png')
    print(model.summary())
    print(model.inputs)
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=lines.output, name=name) 
    print(prediction_model.summary())
    keras.utils.plot_model(prediction_model, to_file='prediction_model.png')
    
    return prediction_model
