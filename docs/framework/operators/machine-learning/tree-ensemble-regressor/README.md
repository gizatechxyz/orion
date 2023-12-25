# Tree Ensemble Regressor

`TreeEnsembleRegressorTrait` provides a trait definition for tree ensemble regressor problem.

```rust
use orion::operators::ml::TreeEnsembleRegressorTrait;
```

### Data types

Orion supports currently only fixed point data types for `TreeEnsembleRegressorTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`tree_ensemble_regressor.predict`](tree_ensemble_regressor.predict.md) | Returns the regressed values for each input in N. |

