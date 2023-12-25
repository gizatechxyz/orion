# Linear Regressor

`LinearRegressorTrait` provides a trait definition for linear regression problem.

```rust
use orion::operators::ml::LinearRegressorTrait;
```

### Data types

Orion supports currently only fixed point data types for `LinearRegressorTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `LinearRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`linear_regressor.predict`](linear_regressor.predict.md) | Performs the generalized linear regression evaluation. |

