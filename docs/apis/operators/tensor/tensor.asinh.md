#tensor.asinh

```rust
fn asinh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the inverse hyperbolic sine of each element in the input tensor.

## Args

- `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the inverse hyperbolic sine of the values in the input tensor.

## Example

```rust
fn asinh_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[-2,-1,0, 1, 2]]
// for each value val we calculate inverse hyperbolic sine with the formula:
// ln( x + sqrt( 1 + x^2))
let result = tensor.asinh();
return result;
}
>>> [-1.4436, -0.8814, 0, 0.8814, 1.4436]
```
