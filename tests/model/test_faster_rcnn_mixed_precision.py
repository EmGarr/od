import numpy as np
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized

from kerod.core.standard_fields import BoxField
from kerod.dataset.preprocessing import expand_dims_for_single_batch, preprocess
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from kerod.model.faster_rcnn import FasterRcnnFPNResnet50Caffe, FasterRcnnFPNResnet50Pytorch


def test_build_fpn_resnet50_faster_rcnn():


    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    num_classes = 2
    model = FasterRcnnFPNResnet50Pytorch(num_classes)

    inputs = {
        'image': np.zeros((100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
            BoxField.LABELS: np.array([1])
        }
    }

    x, y = expand_dims_for_single_batch(*preprocess(inputs))

    # model(x)
    x['ground_truths'] = y
    model(x, training=True)
