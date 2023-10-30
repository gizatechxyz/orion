use orion::numbers::{FixedTrait};

#[derive(Copy, Drop)]
struct TreeClassifier<T> {
    left: Option<Box<TreeClassifier<T>>>,
    right: Option<Box<TreeClassifier<T>>>,
    split_feature: usize,
    split_value: T,
    prediction: T,
    class_distribution: Span<
        T
    >, // assuming class labels of type usize (span index), and probability as T.
}

/// Trait
///
/// predict - Given a set of features, predicts the target value using the constructed decision tree.
/// predict_proba - Predicts class probabilities based on feature data.
trait TreeClassifierTrait<T> {
    /// # TreeClassifierTrait::predict
    ///
    /// ```rust 
    ///    fn predict(ref self: TreeClassifier<T>, features: Span<T>) -> T;
    /// ```
    ///
    /// Predicts the target value for a set of features using the provided decision tree.
    /// 
    /// ## Args
    ///
    /// * `self`: A reference to the decision tree used for making the prediction.
    /// * `features`: A span representing the features for which the prediction is to be made.
    ///
    /// ## Returns
    ///
    /// The predicted target value.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::ml::{FP16x16TreeClassifier, TreeClassifierTrait, TreeClassifier};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn tree_classifier_example(tree: TreeClassifier<FP16x16>) {
    ///
    ///  tree.predict(
    ///        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
    ///    );
    ///    
    /// }
    /// ```
    ///
    fn predict(ref self: TreeClassifier<T>, features: Span<T>) -> T;
    /// # TreeClassifierTrait::predict_proba
    ///
    /// ```rust 
    ///    fn predict_proba(ref self: TreeClassifier<T>, features: Span<T>) -> Span<T>;
    /// ```
    ///
    /// Given a set of features, this method traverses the decision tree
    /// represented by `self` and returns the class distribution (probabilities)
    /// found in the leaf node that matches the provided features. The traversal
    /// stops once a leaf node is reached in the decision tree.
    /// 
    /// ## Args
    ///
    /// * `self`: A reference to the decision tree used for making the prediction.
    /// * `features`: A span representing the features for which the prediction is to be made.
    ///
    /// ## Returns
    ///
    /// Returns a `Span<T>` representing the class distribution at the leaf node.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed points.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::ml::{FP16x16TreeClassifier, TreeClassifierTrait, TreeClassifier};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn tree_classifier_example(tree: TreeClassifier<FP16x16>) {
    ///
    ///  tree.predict_proba(
    ///        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
    ///    );
    ///    
    /// }
    /// ```
    ///
    fn predict_proba(ref self: TreeClassifier<T>, features: Span<T>) -> Span<T>;
}

fn predict<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    ref self: TreeClassifier<T>, features: Span<T>
) -> T {
    let mut current_node: TreeClassifier<T> = self;

    loop {
        match current_node.left {
            Option::Some(left) => {
                match current_node.right {
                    Option::Some(right) => {
                        if *features.at(current_node.split_feature) < current_node.split_value {
                            current_node = left.unbox();
                        } else {
                            current_node = right.unbox();
                        }
                    },
                    Option::None(_) => { break; }
                }
            },
            Option::None(_) => { break; }
        };
    };

    current_node.prediction
}

fn predict_proba<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    ref self: TreeClassifier<T>, features: Span<T>
) -> Span<T> {
    let mut current_node: TreeClassifier<T> = self;

    loop {
        match current_node.left {
            Option::Some(left) => {
                match current_node.right {
                    Option::Some(right) => {
                        if *features.at(current_node.split_feature) < current_node.split_value {
                            current_node = left.unbox();
                        } else {
                            current_node = right.unbox();
                        }
                    },
                    Option::None(_) => { break; }
                }
            },
            Option::None(_) => { break; }
        };
    };

    current_node.class_distribution
}
