# SVM Classifier

`SVMClassifierTrait` provides a trait definition for svm classification problem.

```rust
use orion::operators::ml::SVMClassifierTrait;
```

### Data types

Orion supports currently only fixed point data types for `SVMClassifierTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `SVMClassifierTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`svm_classifier.predict`](svm_classifier.predict.md) | Returns the top class for each of N inputs. |

