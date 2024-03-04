## tensor.reduce_log_sum_exp 

```rust 
   fn reduce_log_sum_exp(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>; 
```

Computes the log sum of the exponentials of the input tensor's elements along the provided axes. 

## Args 
* 'self'(`@Tensor<T>`) - The input tensor.
* 'axis'(`usize`) - The dimension to reduce.
* 'keepdims'(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns 

Returns a new `Tensor<T>` instance with the specified axis reduced by summing its elements.


## Example

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP32x32Tensor;
use orion::numbers::{FixedTrait, FP32x32};

fn reduce_log_sum_exp() -> Tensor<FP32x32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP32x32 { mag: 4294967296, sign: false });
    data.append(FP32x32 { mag: 8589934592, sign: false });
    data.append(FP32x32 { mag: 12884901888, sign: false });
    data.append(FP32x32 { mag: 17179869184, sign: false });
    data.append(FP32x32 { mag: 21474836480, sign: false });
    data.append(FP32x32 { mag: 25769803776, sign: false });
    data.append(FP32x32 { mag: 30064771072, sign: false });
    data.append(FP32x32 { mag: 34359738368, sign: false });
    data.append(FP32x32 { mag: 38654705664, sign: false });
    data.append(FP32x32 { mag: 42949672960, sign: false });
    data.append(FP32x32 { mag: 47244640256, sign: false });
    data.append(FP32x32 { mag: 51539607552, sign: false });
    TensorTrait::new(shape.span(), data.span())

    let tensor = TensorTrait::<FP32x32>::new(shape.span(), data.span());

    return tensor.reduce_log_sum_exp(axis: 2, keepdims: false);

 }   
 
   
>>> [[9215828, 16323477, 20115004], [22716772, 24699744, 26302432]]
``` 
