# tensor.sequence_length

```rust
   fn sequence_length(self: Array<Tensor<T>>) -> Tensor<u32>;
```

Returns the length of the input sequence.

## Args

* `self`(`Array<Tensor<T>>`) - The input sequence.

## Returns

The length of the sequence as scalar, i.e. a tensor of shape [].

## Examples

Let's create new u32 Tensor with constant 42.

```rust
let mut sequence = ArrayTrait::new();

let mut shape = ArrayTrait::<usize>::new();
shape.append(1);
shape.append(2);

let mut data = ArrayTrait::new();
data.append(3);
data.append(1);

sequence.append(TensorTrait::new(shape.span(), data.span()));

sequence.sequence_length()
>>> [1]
```
