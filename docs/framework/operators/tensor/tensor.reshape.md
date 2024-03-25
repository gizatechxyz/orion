# tensor.reshape

```rust 
   fn reshape(self: @Tensor<T>, target_shape: Span<i32>, allowzero: bool) -> Tensor<T>;
```

Reshape the input tensor similar to numpy.reshape. First input is the data tensor, second 
input is a shape tensor which specifies the output shape. It outputs the reshaped tensor. 
At most one dimension of the new shape can be -1. In this case, the value is inferred from 
the size of the tensor and the remaining dimensions. A dimension could also be 0, in which case 
the actual dimension value is unchanged (i.e. taken from the input tensor). If 'allowzero' is set,
and the new shape includes 0, the dimension will be set explicitly to zero (i.e. not taken from input tensor)

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `target_shape`(Span<i32>) - A span containing the target shape of the tensor.
* `allowzero`(`bool`) - Indicates that if any value in the 'shape' input is set to zero, the zero value is honored, similar to NumPy.

## Panics

* Panics if the target shape is incompatible with the input tensor's data.

## Returns

A new `Tensor<T>` with the specified target shape and the same data.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn reshape_tensor_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `reshape` function as follows.
    return tensor.reshape(target_shape: array![2, 4].span(), false);
}
>>> [[0,1,2,3], [4,5,6,7]]
```
