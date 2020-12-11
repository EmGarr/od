# Kerod Model Zoo and Baselines

## Training

- 1x corresponds to : Model trained with this scheduling are heavily under-trained but can be used for fast experiment.

```python
base_lr = 0.02
optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
callbacks = WarmupLearningRateScheduler(base_lr, 1, epochs=[8, 10], init_lr=0.0001),
num_epochs =  12
```

- 3x corresponds to :

```python
base_lr = 0.02
optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
callbacks = WarmupLearningRateScheduler(base_lr, 1, epochs=[28, 34], init_lr=0.0001),
num_epochs =  37
```

### Kerod compared to the other repository

Comparison between 2 of the most famous repositories:

- [TensorPack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) 
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

| Backbone              | training | mAP (box;mask) | TensorPack mAP | Detectron2 mAP |
|-----------------------|----------|----------------|----------------|----------------|
| FasterRcnnFPNResnet50 |       1x | 38             | 37.5           | 37.9           |
| FasterRcnnFPNResnet50 |       3x | WIP            | x              | 40.2           |
| MaskRcnnFPNResnet50   |       1x | WIP            | 38.9;35.4      | 38.6;35.2      |
| MaskRcnnFPNResnet50   |       3x | WIP            |                | 41.0;37.2      |

For comparison [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) best model is a FPN-Resnet101 (training with mask) with a [mAP of 34.7](https://github.com/matterport/Mask_RCNN/issues/1#issuecomment-346984047)
Matterport best model.
