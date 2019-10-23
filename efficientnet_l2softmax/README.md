# Custom implementation of EfficientNet with L2-constrained Softmax

Modified from original implementation
- change input channels from 3 to 1 (grayscale)
- Remove final relu function
- Replace final fc layer to L2-constrained softmax

Currently, support only 1-channel grayscale input.

## usage

```
from efficientnet_l2softmax import EfficientNetL2Softmax
from efficientnet_l2softmax import EfficientNetL2Softmax
model = EfficientNetL2Softmax.from_name(args.model, override_params={'num_classes': 4212, 'image_size': 190})
```

## References

- [https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)


## LICENSE

MIT

