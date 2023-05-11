# tensor.reshape

Returns a new tensor with the specified target shape and the same data as the input tensor.

```rust
fn reshape(self: @Tensor<T>, target_shape: Span<usize>) -> Tensor<T>;
```

#### Args

| Name           | Type          | Description                                       |
| -------------- | ------------- | ------------------------------------------------- |
| `self`         | `@Tensor<T>`  | The input tensor.                                 |
| `target_shape` | `Span<usize>` | A span containing the target shape of the tensor. |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                                |
| ------------------------------------------------------------------------ |
| Panics if the target shape is incompatible with the input tensor's data. |

#### Returns

A new `Tensor<T>` with the specified target shape and the same data.

> _`<T>` generic type depends on Tensor dtype._

#### Examples

```rust
fn reshape_tensor_example() -> Tensor<u32> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,5],[6,7]]]
    let tensor = u32_tensor_2x2x2_helper();
    
    // We set the target shape.
    let mut new_shape = ArrayTrait::new();
    new_shape.append(2);
    new_shape.append(4);
		
    // We can call `reshape` function as follows.
    return tensor.reshape(new_shape.span());
}
>>> [[0,1,2,3], [4,5,6,7]]
```
