# Linear Classifier

`LinearClassifierTrait` provides a trait definition for linear classification problem.

```rust
use orion::operators::ml::LinearClassificationTrait;
```

### Data types

Orion supports currently only fixed point data types for `LinearClassificationTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `LinearClassifierTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`linear_classifier.predict`](linear_classifier.predict.md) | Performs the linear classification evaluation. |
