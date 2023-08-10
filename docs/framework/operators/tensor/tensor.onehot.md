# tensor.onehot

```rust 
   fn onehot(self: @Tensor<T>, depth: usize, axis: Option<usize>, values: Span<usize>) -> Tensor<usize>;
```

Produces one-hot tensor based on input.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `depth`(`usize`) - Scalar or Rank 1 tensor containing exactly one element, specifying the number of classes in one-hot tensor.
* `axis`(`Option<bool>`) - Axis along which one-hot representation in added. Default: axis=-1.
* `values`(`Span<usize>`) - Rank 1 tensor containing exactly two elements, in the format [off_value, on_value]   

## Panics

* Panics if values is not equal to 2.

## Returns 

A new `Tensor<T>` one-hot encode of the input tensor.

## Example

```rust
fn onehot_example() -> Tensor<FixedType> {
    // We instantiate a 1D Tensor here.
    // [[0,1],[2,3]]
    let tensor = u32_tensor_1x3_helper();
    let mut values = ArrayTrait::new();
    values.append(0);
    values.append(1);
    let depth = 3;
    let axis: Option<usize> = Option::None(());
    result = tensor.onehot(depth:depth, axis:axis, values:values.span());
    return result;
}
>>> [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
```
