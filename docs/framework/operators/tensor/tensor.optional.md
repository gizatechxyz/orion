# tensor.optional

```rust 
   fn optional(self: @Tensor<T>) -> Option<Tensor<T>>;
```

Constructs an optional-type value containing either an empty optional of a certain 
type specified by the attribute, or a non-empty value containing the input element.

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

The optional output enclosing the input element.

## Examples

```rust
use core::option::OptionTrait;
fn optional_example() -> Option<Tensor<T>> {
    let a = TensorTrait::<
        FP16x16
    >::new(
        shape: array![4, 2].span(),
        data: array![
           1_i8,
           2_i8,
           3_i8,
           4_i8,
           5_i8,
           6_i8,
           7_i8,
           8_i8
        ].span(),
    );
    a.optional()
}
>>> Option[Tensor[1,2,3,4,5,6,7,8]]
    
```
