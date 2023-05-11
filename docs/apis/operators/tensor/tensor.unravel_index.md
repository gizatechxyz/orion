# tensor.unravel\_index

Converts a one-dimensional index to a multi-dimensional index.

```rust
fn unravel_index(self: @Tensor<T>, index: usize) -> Span<usize>;
```

#### Args

| Name      | Type          | Description           |
| --------- | ------------- | --------------------- |
| `self`    | `@Tensor<T>`  | The input tensor.     |
| `indices` | `Span<usize>` | The index to unravel. |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                 |
| --------------------------------------------------------- |
| Panics if the index is out of bounds of the Tensor shape. |

#### Returns

The unraveled indices corresponding to the given index.

#### Examples

```rust
fn unravel_index_example() -> Span<usize> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();
		
    // We can call `unravel_index` function as follows.
    return tensor.unravel_index(3);
}
>>> [0,1,1] 
// This means that the value of index 3 of Tensor.data
// can be found at indices [0,1,1] in multidimensional representation.
```
