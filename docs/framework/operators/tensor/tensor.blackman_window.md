# tensor.blackman_window

```rust 
   fn blackman_window(size: T, periodic: Option<usize>) -> Tensor<T>;
```

Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.


* `size`(`T`) - A scalar value indicating the length of the window.
* `periodic`(Option<usize>) - If 1, returns a window to be used as periodic function. If 0, return a symmetric window. When 'periodic' is specified, hann computes a window of length size + 1 and returns the first size points. The default value is 1.

## Returns

A Blackman window with length: size. The output has the shape: [size].

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::{FixedTrait, FP8x23};


fn blackman_window_example() -> Tensor<FP8x23> {
    return TensorTrait::blackman_window(FP8x23 { mag: 33554432, sign: false }, Option::Some(0));  // size: 4
}
>>> [0  0.36  0.36  0]
```
