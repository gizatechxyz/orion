# Tree Ensemble 

`TreeEnsembleTrait` provides a trait definition for tree ensemble problem.

```rust
use orion::operators::ml::TreeEnsembleTrait;
```

### Data types

Orion supports currently only fixed point data types for `TreeEnsembleTrait`.

| Data type            | dtype                                                         |
| -------------------- | ------------------------------------------------------------- |
| Fixed point (signed) | `TreeEnsembleTrait<FP8x23 \| FP16x16 \| FP64x64 \| FP32x32>` |


***

| function | description |
| --- | --- |
| [`tree_ensemble.predict`](tree_ensemble.predict.md) | Returns the regressed values for each input in a batch. |