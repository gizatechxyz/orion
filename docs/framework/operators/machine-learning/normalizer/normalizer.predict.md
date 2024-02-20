# Normalizer::predict

```rust 
   fn predict(X: Tensor<T>, norm: NORM) -> Tensor<T>;
```

Returns the normalized input.
Tree different types of normalization can be performed and are defined as follow :
MAX: $Y = \frac{X}{max(X)}$
L1: $Y = \frac{X}{sum(X)}$
L2: $Y = \frac{X}\sqrt{{sum(X²)}}$
For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row of the batch is normalized independently.
    
## Args

* `X`(`@Tensor<T>`) - Input 2D tensor. 
* `norm`(`NORM`) - NORM::MAX, NORM::L1 or NORM::L2


## Returns

* Tensor<T> - output tensor 

## Examples

```rust
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorPartialEq};

use orion::operators::ml::normalizer::normalizer::{
    NormalizerTrait, NORM
};



fn normalizer_max() ->  Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

  let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 52428, sign: true });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 39321, sign: false });

  let X = TensorTrait::new(shape.span(), data.span());

  return NormalizerTrait::predict(X, NORM::MAX);
}
>>> [[-1.        -0.8       -0.6      ]
     [-1.        -0.5        0.       ]
     [ 0.3333333  0.6666666  1.       ]]

```

