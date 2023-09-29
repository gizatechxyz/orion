# TreeRegressorTrait::fit

```rust 
   fn fit(data: Span<Span<T>>, target: Span<T>, max_depth: usize, random_state: usize) -> TreeNode<T>;
```

Builds a decision tree based on the provided data and target values up to a specified maximum depth.

## Args

* `data`: A span of spans representing rows of features in the dataset.
* `target`: A span representing the target values corresponding to each row in the dataset.
* `max_depth`: The maximum depth of the decision tree. The tree stops growing once this depth is reached.
* `random_state`: It ensures that the tie-breaking is consistent across multiple runs, leading to reproducible results.

## Returns

A `TreeNode` representing the root of the constructed decision tree.

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

 TreeRegressorTrait::fit(data, target, 3, 42);
}
```
