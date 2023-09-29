# tree.predict

```rust
   fn predict(ref self: TreeNode<T>, features: Span<T>) -> T;
```

Predicts the target value for a set of features using the provided decision tree.

## Args

* `self`: A reference to the decision tree used for making the prediction.
* `features`: A span representing the features for which the prediction is to be made.

## Returns

The predicted target value.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use orion::operators::ml::{FP16x16TreeRegressor, TreeRegressorTrait};
use orion::numbers::{FP16x16, FixedTrait};

fn tree_regressor_example() {

 let data = array![
     array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span(),
     array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span(),
     array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span(),
     array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span(),
 ]
     .span();

 let target = array![
     FixedTrait::new_unscaled(2, false),
     FixedTrait::new_unscaled(4, false),
     FixedTrait::new_unscaled(6, false),
     FixedTrait::new_unscaled(8, false)
 ]
     .span();

 let mut tree = TreeRegressorTrait::fit(data, target, 3);

 let prediction_1 = tree
   .predict(
       array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
   );
}
```
