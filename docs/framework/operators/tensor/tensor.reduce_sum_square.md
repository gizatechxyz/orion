## tensor.reduce_sum_square

```rust 
   fn reduce_sum_square(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Computes the sum square of the input tensor's elements along the provided axes. 
## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The dimension to reduce.
* `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.

fn reduce_sum_square_example() -> Tensor<u32> {

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    let tensor = TensorTrait::<u32>::new(shape.span(), data.span());

    We can call `reduce_sum_square` function as follows.
    return tensor.reduce_sum_square(axis: 1, keepdims: true);
}
>>> [[5, 25]]
```
