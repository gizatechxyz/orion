# Normalizer

`NormalizerTrait` computes the normalization of the input, each row of the input is normalized independently.

```rust
use orion::operators::ml::NormalizerTrait;
```

### Data types

Orion supports currently only fixed point data types for `NormalizerTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `NormalizerTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`normalizer.predict`](normalizer.predict.md) | Returns the normalization of the input, each row of the input is normalized independently. |

