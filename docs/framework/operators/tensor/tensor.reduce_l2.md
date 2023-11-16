## tensor.reduce_l2

```rust 
   fn reduce_l2(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Computes the L2 norm of the input tensor's elements along the provided axes.
## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The dimension to reduce.
* `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.

fn reduce_l2_example() -> Tensor<u32> {

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(5, false));
    let tensor = TensorTrait::<FP8x23>::new(shape.span(), data.span());

    We can call `reduce_l2` function as follows.
    return tensor.reduce_l2(axis: 1, keepdims: true);
}
>>> [[0x11e3779, 0x2ea5ca1]]
```
