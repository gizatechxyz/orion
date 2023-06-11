# tensor.ln

```rust
fn ln(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the natural log of all elements of the input tensor.
$$
y_i=ln({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the natural log of the elements of the input tensor.

## Examples

```rust
fn ln_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// [[1,2,3,100]]
let mut sizes = ArrayTrait::new();
sizes.append(4);

let mut data = ArrayTrait::new();
data.append(IntegerTrait::new(1_u32, false));
data.append(IntegerTrait::new(2_u32, false));
data.append(IntegerTrait::new(3_u32, false));
data.append(IntegerTrait::new(100_u32, false));
let extra = Option::<ExtraParams>::None(());
let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra)
/// Note that we cannot use the helper tensor since it includes a 0 value
/// and we cannot take the natural log ln(0) since it's undefined.


// We can call `ln` function as follows.
return tensor.ln();
}
>>> [[0, 5814538, 9215825, 38630966]]
// The fixed point representation of
/// [[0, 0.693147, 1.098612, 4.605170]]
```
