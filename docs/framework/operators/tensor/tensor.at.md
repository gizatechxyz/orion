# tensor.at

```rust 
   fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
```

Retrieves the value at the specified indices of a Tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Span<usize>`) - The indices to access element of the Tensor.

## Panics

* Panics if the number of indices provided don't match the number of dimensions in the tensor.

## Returns

The `T` value at the specified indices.

# Examples

```rust
fn at_example() -> u32 {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();
    
    // We set indices to access element of the Tensor.
    let mut indices = ArrayTrait::new();
    indices.append(0);
    indices.append(1);
    indices.append(1);
		
    // We can call `at` function as follows.
    return tensor.at(indices.span());
}
>>> 3
```
