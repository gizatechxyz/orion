# Sequence

A Sequence represents an array of tensors.

```rust
use orion::operators::sequence;
```

### Data types

Orion supports currently these `Sequence` types.

| Data type                 | dtype                                                    |
| ------------------------- | -------------------------------------------------------- |
| 32-bit integer (signed)   | `Array<Tensor<i32>>`                                     |
| 8-bit integer (signed)    | `Array<Tensor<i8>>`                                      |
| 32-bit integer (unsigned) | `Array<Tensor<u32>>`                                     |
| Fixed point (signed)      | `Array<Tensor<FP8x23 \| FP16x16 \| FP32x32 \| FP64x64>>` |

### Sequence**Trait**

`SequenceTrait` defines the operations that can be performed on a Sequence of tensors.

| function | description |
| --- | --- |
| [`sequence.sequence_construct`](sequence.sequence\_construct.md) | Constructs a tensor sequence containing the input tensors. |
| [`sequence.sequence_empty`](sequence.sequence\_empty.md) | Returns an empty tensor sequence. |

