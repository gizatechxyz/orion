## sequence.sequence_empty

```rust
   fn sequence_empty() -> Array<Tensor<T>>;
```

Returns an empty tensor sequence.

## Args

## Returns

An empty `Array<Tensor<T>>` instance.

## Examples

Let's create a new empty sequence.

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{
    TensorTrait, // we import the trait
    Tensor, // we import the type
    U32Tensor // we import the implementation. 
};
use orion::operators::sequence::SequenceTrait;

fn sequence_empty_example() -> Array<Tensor<u32>> {
    let sequence = SequenceTrait::sequence_empty();

    return sequence;
}

>>> []
```
