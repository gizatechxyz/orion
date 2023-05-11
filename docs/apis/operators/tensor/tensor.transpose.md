# tensor.transpose

Returns a new tensor with the axes rearranged according to the given permutation.

```rust
fn transpose(self: @Tensor<T>, axes: Span<usize>) -> Tensor<T>;
```

#### Args

| Name   | Type          | Description                                                |
| ------ | ------------- | ---------------------------------------------------------- |
| `self` | `@Tensor<T>`  | The input tensor.                                          |
| `axes` | `Span<usize>` | The usize elements representing the axes to be transposed. |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                                            |
| ------------------------------------------------------------------------------------ |
| Panics if the length of the axes array is not equal to the rank of the input tensor. |

#### Returns

A `Tensor<T>` instance with the axes reordered according to the given permutation.

> _`<T>` generic type depends on Tensor dtype._

#### Examples

```rust
fn transpose_tensor_example() -> Tensor<u32> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();

    // We set the axes to be transposed.
    let mut axes = ArrayTrait::new();
    axes.append(1);
    axes.append(2);
    axes.append(0);
		
    // We can call `transpose` function as follows.
    return tensor.transpose(axes.span());
}
>>> [[[0,4],[1,5]],[[2,6],[3,7]]]
```
