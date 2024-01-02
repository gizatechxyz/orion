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
   use orion::operators::tensor::FP8x23Tensor;
   use orion::numbers::{FixedTrait, FP8x23};

   fn input_0() -> Tensor<FP8x23> {
       let mut shape = ArrayTrait::<usize>::new();
       shape.append(3);
       shape.append(2);
       shape.append(2);

   let mut data = ArrayTrait::new();
   data.append(FixedTrait::new_unscaled(1, false));
   data.append(FixedTrait::new_unscaled(2, false));
   data.append(FixedTrait::new_unscaled(3, false));
   data.append(FixedTrait::new_unscaled(4, false));
   data.append(FixedTrait::new_unscaled(5, false));
   data.append(FixedTrait::new_unscaled(6, false));
   data.append(FixedTrait::new_unscaled(7, false));
   data.append(FixedTrait::new_unscaled(8, false));
   data.append(FixedTrait::new_unscaled(9, false));
   data.append(FixedTrait::new_unscaled(10, false));
   data.append(FixedTrait::new_unscaled(11, false));
   data.append(FixedTrait::new_unscaled(12, false));

>>> (
      [[[1, 2]   
        [3, 4]]    
      
       [[5, 6]]
        [7, 8]]

       [[9, 10]
        [11, 12]]]
   )

         
   let tensor = TensorTrait::new(shape.span(), data.span())
   
   return tensor.reduce_log_sum_exp(axis: 2, keepdims: false);
}

>>> 
    (
        [[2.31, 4.31]
         [6.31, 8.31]
         [10.31, 12.31]]
    )

``` 