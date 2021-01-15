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

import tensorflow
from tensorflow import keras


def center_alpha_transform_inv(center_alphas, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to center_alphas (usually anchors).

    Before applying the deltas to the center_alphas, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the center_alphas.

    Args
        center_alphas : np.array of shape (B, N, 3), where B is the batch size, N the number of center_alphas and 4 values for (x, y, alpha).
        deltas: np.array of same shape as center_alphas. These deltas (d_x, d_y, d_alpha).
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as center_alphas, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2]

    x = center_alphas[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0])
    y = center_alphas[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1])
    alpha = center_alphas[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2])

    pred_center_alphas = keras.backend.stack([x, y, alpha], axis=2)

    return pred_center_alphas


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tensorflow.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def map_fn(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/map_fn .
    """

    if "shapes" in kwargs:
        shapes = kwargs.pop("shapes")
        dtype = kwargs.pop("dtype")
        sig = [tensorflow.TensorSpec(shapes[i], dtype=t) for i, t in
               enumerate(dtype)]

        # Try to use the new feature fn_output_signature in TF 2.3, use fallback if this is not available
        try:
            return tensorflow.map_fn(*args, **kwargs, fn_output_signature=sig)
        except TypeError:
            kwargs["dtype"] = dtype

    return tensorflow.map_fn(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.compat.v1.image.resize_images(images, size, methods[method], align_corners)
