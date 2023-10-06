# Tree Regressor

`TreeRegressorTrait` provides a trait definition for decision tree regression. This trait offers functionalities to build a decision tree and predict target values based on input features.

```rust
use orion::operators::ml::TreeRegressorTrait;
```

### Data types

Orion supports currently only fixed point data types for `TreeRegressorTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |

***

| function | description |
| --- | --- |
| [`tree.fit`](tree.fit.md) | Constructs a decision tree regressor based on the provided data and target values. |
| [`tree.predict`](tree.predict.md) | Given a set of features, predicts the target value using the constructed decision tree. |

