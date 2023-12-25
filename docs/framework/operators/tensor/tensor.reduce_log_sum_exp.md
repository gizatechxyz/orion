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

fn reduce_log_sum_exp() -> Tensor<u32> {

let tensor = TensorTrait::new(
    shape: array![2, 2, 2].span(),
    data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
); 

We can call `reduce_log_sum_exp` function as follows.

return tensor.reduce_log_sum_exp(axis: 2, keepdims: false);
>>> 