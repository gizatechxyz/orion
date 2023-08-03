# tensor.log

```rust 
    fn log(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the natural log of all elements of the input tensor.
$$
y_i=log({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the natural log of the elements of the input tensor.

## Examples

```rust
fn log_example() -> Tensor<FixedType> {
    // We instantiate a 1D Tensor here.
    // [[1,2,3,100]]
    let mut sizes = ArrayTrait::new();
    sizes.append(4);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(1, false));
    data.append(IntegerTrait::new(2, false));
    data.append(IntegerTrait::new(3, false));
    data.append(IntegerTrait::new(100, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra)
		
    // We can call `log` function as follows.
    return tensor.log();
}
>>> [[0, 5814538, 9215825, 38630966]]
// The fixed point representation of
/// [[0, 0.693147, 1.098612, 4.605170]]
```
