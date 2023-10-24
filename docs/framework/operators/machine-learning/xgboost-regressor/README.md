# Tree Regressor

`XGBoostRegressorTrait` provides a trait definition for xgboost regression. This trait offers functionalities to predict target values based on input features.

```rust
use orion::operators::ml::XGBoostRegressorTrait;
```

### Data types

Orion supports currently only fixed point data types for `XGBoostRegressorTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |

***

| function | description |
| --- | --- |
| [`xgboost.predict`](xgboost.predict.md) | Predicts the target value for a set of features using the provided ensemble of decision trees. |

