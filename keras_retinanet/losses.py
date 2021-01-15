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


def focal(alpha=0.25, gamma=2.0, cutoff=0.5):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
        cutoff: Positive prediction cutoff for soft targets

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = tensorflow.where(keras.backend.not_equal(anchor_state, -1))
        labels         = tensorflow.gather_nd(labels, indices)
        classification = tensorflow.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tensorflow.where(keras.backend.greater(labels, cutoff), alpha_factor, 1 - alpha_factor)
        focal_weight = tensorflow.where(keras.backend.greater(labels, cutoff), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tensorflow.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 4). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 3).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        xy_regression        = y_pred[:, :, :2]
        xy_regression_target = y_true[:, :, :2]
        
        # separate target and state
        alpha_regression        = y_pred[:, :, 2]
        alpha_regression_target = y_true[:, :, 2]
        
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = tensorflow.where(keras.backend.equal(anchor_state, 1))
        xy_regression        = tensorflow.gather_nd(xy_regression, indices)
        xy_regression_target = tensorflow.gather_nd(xy_regression_target, indices)
        
        alpha_regression       = tensorflow.gather_nd(alpha_regression, indices)
        alpha_regression_target = tensorflow.gather_nd(alpha_regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        xy_regression_diff = xy_regression - xy_regression_target
        xy_regression_diff = keras.backend.abs(xy_regression_diff)
        
        xy_regression_loss = tensorflow.where(
            keras.backend.less(xy_regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(xy_regression_diff, 2),
            xy_regression_diff - 0.5 / sigma_squared
        )

        # regression_loss_dpi
        ones = keras.backend.ones_like(alpha_regression)
        alpha_regression_loss = ones - keras.backend.cos((alpha_regression_target - alpha_regression) * (3.1415 / 180.0))

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        
        normalized_loss = (keras.backend.sum(xy_regression_loss) + keras.backend.sum(alpha_regression_loss)) / normalizer

        return normalized_loss

    return _smooth_l1
