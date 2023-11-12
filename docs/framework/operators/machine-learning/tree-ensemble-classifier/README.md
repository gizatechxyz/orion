# Tree Ensemble Classifier

`TreeEnsembleClassifierTrait` provides a trait definition for tree ensemble classification problem.

```rust
use orion::operators::ml::TreeEnsembleClassifierTrait;
```

### Data types

Orion supports currently only fixed point data types for `TreeEnsembleClassifierTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeRegressorTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |

### How to construct `TreeEnsembleClassifier` 

You can utilize [this notebook](https://colab.research.google.com/drive/1qem56rUKJcNongXsLZ16_869q8395prz#scrollTo=V3qGW_kfXudk) to translate parameters from your ONNX TreeEnsembleClassifier model into Cairo code. Efforts are underway to integrate this functionality into Giza-CLI, aiming to enhance the user experience.


***

| function | description |
| --- | --- |
| [`tree_ensemble_classifier.predict`](tree_ensemble_classifier.predict.md) | Returns the top class for each of N inputs. |

