# tensor.reduce\_sum

Reduces a tensor by summing its elements along a specified axis.

```rust
fn reduce_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

#### Args

| Name       | Type         | Description                                        |
| ---------- | ------------ | -------------------------------------------------- |
| `self`     | `@Tensor<T>` | The input tensor.                                  |
| `axis`     | `usize`      | The dimension to reduce.                           |
| `keepdims` | `bool`       | If true, retains reduced dimensions with length 1. |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                            |
| -------------------------------------------------------------------- |
| Panics if axis is not in the range of the input tensor's dimensions. |

#### Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.

> _`<T>` generic type depends on Tensor dtype._

#### Examples

```rust
fn reduce_sum_example() -> Tensor<u32> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();
		
    // We can call `reduce_sum` function as follows.
    return tensor.reduce_sum(0, false);
}
>>> [[4,6],[8,10]]
```
