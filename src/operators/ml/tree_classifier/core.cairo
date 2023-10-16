use orion::numbers::{FixedTrait};

#[derive(Copy, Drop)]
struct TreeClassifier<T> {
    left: Option<Box<TreeClassifier<T>>>,
    right: Option<Box<TreeClassifier<T>>>,
    split_feature: usize,
    split_value: T,
    prediction: T,
    class_distribution: Span<T>, // assuming class labels of type usize (span index), and probability as T.
}

/// Trait
///
/// predict - Given a set of features, predicts the target value using the constructed decision tree.
/// predict_proba - Given a set of features, predicts the probability of each X example being of a given class..
trait TreeClassifierTrait<T> {
    fn predict(ref self: TreeClassifier<T>, features: Span<T>) -> T;
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
                    Option::None(_) => {
                        break;
                    }
                }
            },
            Option::None(_) => {
                break;
            }
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
                    Option::None(_) => {
                        break;
                    }
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    current_node.class_distribution
}
