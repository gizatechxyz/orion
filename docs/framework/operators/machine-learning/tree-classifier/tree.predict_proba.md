# TreeClassifierTrait::predict_proba

```rust 
   fn predict_proba(ref self: TreeClassifier<T>, features: Span<T>) -> Span<T>;
```

Given a set of features, this method traverses the decision tree
represented by `self` and returns the class distribution (probabilities)
found in the leaf node that matches the provided features. The traversal
stops once a leaf node is reached in the decision tree.

## Args

* `self`: A reference to the decision tree used for making the prediction.
* `features`: A span representing the features for which the prediction is to be made.

## Returns

Returns a `Span<T>` representing the class distribution at the leaf node.

## Type Constraints

Constrain input and output types to fixed points.

## Examples

```rust
use orion::operators::ml::{FP16x16TreeClassifier, TreeClassifierTrait, TreeClassifier};
use orion::numbers::{FP16x16, FixedTrait};

fn tree_classifier_example(tree: TreeClassifier<FP16x16>) {

 tree.predict_proba(
       array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
   );
   
}
```
