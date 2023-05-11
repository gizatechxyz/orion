# tensor.exp

Computes the exponential of all elements of the input tensor.

$$
y_i=e^{x_i}
$$

```rust
fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
```

#### Args

| Name   | Type         | Description       |
| ------ | ------------ | ----------------- |
| `self` | `@Tensor<T>` | The input tensor. |

> _`<T>` generic type depends on Tensor dtype._

#### Returns

Returns a new tensor in [`FixedType`](../../numbers/fixed-point/) with the exponential of the elements of the input tensor.

#### Examples

```rust
fn exp_example() -> Tensor<FixedType> {
    // We instantiate a 2D Tensor here.
    // [[0,1],[2,3]]
    let tensor = u32_tensor_2x2_helper();
		
    // We can call `exp` function as follows.
    return tensor.exp();
}
>>> [[67108864,182420802],[495871144,1347917552]]
// The fixed point representation of
// [[1, 2.718281],[7.38905, 20.085536]]
```
