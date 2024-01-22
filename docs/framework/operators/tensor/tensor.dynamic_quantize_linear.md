# tensor.dynamic_quantize_linear

```rust
fn dynamic_quantize_linear(self: @Tensor<T>) -> (Tensor::<Q>, Tensor<T>, Tensor<T>);
```

Quantizes a Tensor using dynamic linear quantization.

The dynamic linear quantization operator. It consumes a high precision tensor 
to compute the low precision / quantized tensor dynamicly. 
Right now only uint8 is supported, it saturates to [0, 255].

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<Q>` with the same shape as the input tensor, containing the quantized values.
* `y_scale`(`@Tensor<T>`) - Scale for doing quantization to get `y`.
* `y_zero_point`(`@Tensor<T>`) - Zero point for doing quantization to get `y`.

## Type Constraints

* `T` in (`Tensor<FP>`, `Tensor<i8>`, `Tensor<i32>`, `tensor<u32>`)
* `Q` in (`Tensor<i32>`)- Constrain `y` to 8-bit unsigned integer tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor};
use orion::numbers::{u8, i32, IntegerTrait};

fn dynamic_quantize_linear_example() -> (Tensor<u8>, Tensor<FP16x16>, Tensor<u8>) {
    // We instantiate a 1D Tensor here.
    let x = TensorTrait::<FP16x16>::new(
        shape: array![6].span(),
        data: array![
            FP16x16 { mag: 10945, sign: false },
            FP16x16 { mag: 190054, sign: false },
            FP16x16 { mag: 196608, sign: false },
            FP16x16 { mag: 229376, sign: false },
            FP16x16 { mag: 196608, sign: true },
            FP16x16 { mag: 229376, sign: true },
        ]
            .span(),
    );

    return x.dynamic_quantize_linear();
}
>>> [133, 233, 236, 255, -18, -0]
```
