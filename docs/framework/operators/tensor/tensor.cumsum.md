# tensor.cumsum

```rust 
   fn cumsum(self: @Tensor<T>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>) -> Tensor<usize>;
```

Performs cumulative sum of the input elements along the given axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the cumulative sum.
* `exclusive`(`Option<bool>`) - By default, it will do the sum inclusively meaning the first element is copied as is.
* `reverse`(`Option<bool>`) - If true, the cumulative sum is performed in the opposite direction. Defaults to false.   

## Panics

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns 

A new `Tensor<T>` instance containing the cumulative sum of the input tensor's elements along the given axis.

## Examples

Case 1: cumsum with default parameters

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};

fn cumsum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    return tensor.cumsum(
        axis: 2,
        exclusive: Option::None(()),
        reverse: Option::None(())
    );
}
>>> [[[0,1],[2,5]],[[4,9],[6,13]]]
```

Case 2: cumsum with exclusive = true

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};

fn cumsum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    return tensor.cumsum(
        axis: 2,
        exclusive: Option::Some(true),
        reverse: Option::None(())
    );
}
>>> [[[0,0],[0,2]],[[0,4],[0,6]]]
```

Case 3: cumsum with exclusive = true and reverse = true

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};

fn cumsum_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    return tensor.cumsum(
        axis: 2,
        exclusive: Option::Some(true),
        reverse: Option::Some(true)
    );
}
>>> [[[1,0],[3,0]],[[5,0],[7,0]]]
```
