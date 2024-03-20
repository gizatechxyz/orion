## tensor.expand

```rust 
   fn expand(self: @Tensor<T>, Tensor: Span<usize>,) -> Tensor<T>;
```

Broadcast the input tensor following the given shape and the broadcast rule. The broadcast rule is similar to numpy.array(input) * numpy.ones(shape): Dimensions are right alignment; Two corresponding dimensions must have the same value, or one of them is equal to 1.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `shape`(`Tensor<usize>`) - A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

## Panics 

* If the shape doesn't follow the broadcast rule.

## Returns

A new `Tensor<T>` result of the expansion.

## Examples

```rust
use orion::operators::tensor::{FP16x16Tensor};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{U32Tensor};
use orion::numbers::FP16x16;

fn test_expand() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    return X.expand(TensorTrait::new(array![3].span(),array![2, 1, 6].span()));

}

>>> [[[1. 1. 1. 1. 1. 1.]
      [2. 2. 2. 2. 2. 2.]
      [3. 3. 3. 3. 3. 3.]]
    
     [[1. 1. 1. 1. 1. 1.]
      [2. 2. 2. 2. 2. 2.]
      [3. 3. 3. 3. 3. 3.]]]
```
