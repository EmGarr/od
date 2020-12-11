# Hyperparameters

In many, object detection framework you have a `huge` configuration file. It is not well documented and usually make the code difficult to read.
I decided to not provide any configuration file. I hope the hyperparameters can be found at a logical location. 

If you can't, you need to write code.

## Tensorpack hyperparameters

Tensorpack is the tensorflow repository which provide the best performances. However, it has been designed
for training and not serving (even if the graph is fully servable). There is a big gap between the performances
of tensorpack and the tensorflow object detection library. We took all the hyperparameters that were
different in the repository.

### Box encoding

They advice and set the box_encoding to [10.0, 10.0, 5.0, 5.0](https://github.com/osrf/tensorflow_object_detector/blob/master/src/object_detection/builders/box_coder_builder.py#L40). It isn't the case in tensorpack.
In this implementation, we don't use the scale factors.

### FastRCNN Head

Tensopack use an L1 instead of a smooth L1


### RPN loss 

delta set to 1 / 9 and the loss coeff to 9 

### Groundtruths as ROI

To know this one you need to read the code multiple times of detectron are tensorpack (the two repositories). You need to add your ground_truths as ROI during the sampling.

[] in progress

