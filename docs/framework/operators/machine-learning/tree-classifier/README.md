# Tree Classifier

`TreeClassifierTrait` provides a trait definition for decision tree classifier. This trait offers functionalities to build a decision tree and predict target values based on input features.

```rust
use orion::operators::ml::TreeClassifierTrait;
```

### Data types

Orion supports currently only fixed point data types for `TreeClassifierTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeClassifierTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |

***

| function | description |
| --- | --- |
| [`tree.predict`](tree.predict.md) | Given a set of features, predicts the target value using the constructed decision tree. |
| [`tree.predict_proba`](tree.predict\_proba.md) | Predicts class probabilities based on feature data. |

