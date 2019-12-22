import tensorflow as tf

from od.core.standard_fields import BoxField, DatasetField


def resize_to_min_dim(image, short_edge_length, max_dimension):
    """Resize an image given to the min size maintaining the aspect ratio.

    If one of the image dimensions is bigger than the max_dimension after resizing, it will scale
    the image such that its biggest dimension is equal to the max_dimension.

    Arguments :

    - *image*: A np.array of size [height, width, channels].
    - *short_edge_length*: minimum image dimension.
    - *max_dimension*: If the resized largest size is over max_dimension. Will use to max_dimension
    to compute the resizing ratio.

    Returns:
    - *resized_image*: The input image resized with the aspect_ratio preserved in float32
    """
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    im_size_min = tf.minimum(height, width)
    im_size_max = tf.maximum(height, width)
    scale = short_edge_length / im_size_min
    # Prevent the biggest axis from being more than MAX_SIZE
    if tf.math.round(scale * im_size_max) > max_dimension:
        scale = max_dimension / im_size_max

    target_height = tf.cast(height * scale, dtype=tf.int32)
    target_width = tf.cast(width * scale, dtype=tf.int32)
    return tf.image.resize(tf.expand_dims(image, axis=0),
                           size=[target_height, target_width],
                           method=tf.image.ResizeMethod.BILINEAR)[0]


@tf.function
def preprocess(inputs):
    """This operations performs a classical preprocessing operations for localization datasets:

    - COCO
    - Pascal Voc

    You can download easily those dataset using [tensorflow dataset](https://www.tensorflow.org/datasets/catalog/overview).

    Argument:

    - *inputs*: It can be either a [FeaturesDict](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict) or a dict.
    but it should have the following structures.

    ```python
    inputs = FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'objects': Sequence({
            'area': Tensor(shape=(), dtype=tf.int64), # area
            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32), # The values are between 0 and 1
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=80),
        }),
    })
    ```

    Returns:

    - *inputs*:
        1. image: A 3D tensor of float32 and shape [None, None, 3]
        2. image_information: A 1D tensor of float32 and shape [(height, width),]. It contains the shape
        of the image without any padding. It can be usefull if it followed by a `padded_batch` operations.
        The models needs those information in order to clip the boxes to the proper dimension.
    - *inputs*: A dict with the following information

    ```
    inputs = {
        BoxField.BOXES: A tensor of shape [num_boxes, (y1, x1, y2, x2)] and resized to the image shape
        BoxField.LABELS: A tensor of shape [num_boxes, ]
        BoxField.NUM_BOXES: A tensor of shape (). It is usefull to unpad the data in case of a batched training
    }
    ```
    """
    image = resize_to_min_dim(inputs['image'], 800.0, 1300.0)
    image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    boxes = inputs['objects'][BoxField.BOXES] * tf.tile(tf.expand_dims(image_information, axis=0),
                                                        [1, 2])
    return {
        DatasetField.IMAGES: image,
        DatasetField.IMAGES_INFO: image_information,
        BoxField.BOXES: boxes,
        BoxField.LABELS: inputs['objects'][BoxField.LABELS],
        BoxField.NUM_BOXES: tf.shape(inputs['objects'][BoxField.LABELS]),
        BoxField.WEIGHTS: tf.fill(tf.shape(inputs['objects'][BoxField.LABELS]), 1.0)
    }


def expand_dims_for_single_batch(inputs):
    """In order to train your model you need to add a batch dimension to the output of the preprocess
    function. For a single batch operation this method is faster:

    - `expand_dims`:

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ```

    > Execution time: 0.002636657891998766

    - `batch`

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(1)
    ```

    > Execution time: 0.004332915792008862

    - `padded_batch`

    ```python
    ds_train = tfds.load(name="voc", split="train", shuffle_files=True)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.padded_batch(batch_size, padded_shapes=padded_shapes)
    ```

    > Execution time: 0.0055130551019974515

    Returns:

    - *inputs*:
        1. image: A 3D tensor of float32 and shape [None, None, 3]
        2. image_information: A 1D tensor of float32 and shape [(height, width),]. It contains the shape
        of the image without any padding. It can be usefull if it followed by a `padded_batch` operations.
        The models needs those information in order to clip the boxes to the proper dimension.

    - *inputs*: A dict with the following information

    ```
    inputs = {
        BoxField.BOXES: A tensor of shape [1, num_boxes, (y1, x1, y2, x2)] and resized to the image shape
        BoxField.LABELS: A tensor of shape [1, num_boxes, ]
        BoxField.NUM_BOXES: A tensor of shape [1, 1]. It is usefull to unpad the data in case of a batched training
    }
    ```
    """
    return {
        DatasetField.IMAGES: tf.expand_dims(inputs[DatasetField.IMAGES], axis=0),
        DatasetField.IMAGES_INFO: tf.expand_dims(inputs[DatasetField.IMAGES_INFO], axis=0),
        BoxField.BOXES: tf.expand_dims(inputs[BoxField.BOXES], axis=0),
        BoxField.LABELS: tf.expand_dims(inputs[BoxField.LABELS], axis=0),
        BoxField.NUM_BOXES: tf.expand_dims(inputs[BoxField.NUM_BOXES], axis=0),
        BoxField.WEIGHTS: tf.expand_dims(inputs[BoxField.WEIGHTS], axis=0)
    }