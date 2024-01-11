# SVM Regressor

`SVMRegressorTrait` provides a trait definition for svm regression problem.

```rust
use orion::operators::ml::SVMRegressorTrait;
```

### Data types

Orion supports currently only fixed point data types for `SVMRegressorTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `SVMRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`svm_regressor.predict`](svm_regressor.predict.md) | Returns the regressed values for each input in N. |

