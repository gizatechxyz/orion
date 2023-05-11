# tensor.ravel\_index

Converts a multi-dimensional index to a one-dimensional index.

```rust
fn ravel_index(self: @Tensor<T>, indices: Span<usize>) -> usize;
```

#### Args

| Name      | Type          | Description                         |
| --------- | ------------- | ----------------------------------- |
| `self`    | `@Tensor<T>`  | The input tensor.                   |
| `indices` | `Span<usize>` | The indices of the Tensor to ravel. |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                    |
| ------------------------------------------------------------ |
| Panics if the indices are out of bounds of the Tensor shape. |

#### Returns

The index corresponding to the given indices.

#### Examples

```rust
fn ravel_index_example() -> usize {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();
    
    // We set the indices of the Tensor to ravel.
    let mut indices = ArrayTrait::new();
    indices.append(1);
    indices.append(3);
    indices.append(0);
		
    // We can call `ravel_index` function as follows.
    return tensor.ravel_index(indices.span());
}
>>> 10 
// This means that the value of indices [1,3,0] 
// of a multidimensional array can be found at index 10 of Tensor.data.
```
