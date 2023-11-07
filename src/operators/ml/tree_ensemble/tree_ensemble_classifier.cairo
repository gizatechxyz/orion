use core::nullable::NullableTrait;
use core::dict::Felt252DictTrait;
use core::dict::Felt252DictEntryTrait;
use nullable::{match_nullable, FromNullableResult};

use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::ml::tree_ensemble::core::{TreeEnsemble, TreeEnsembleImpl, TreeEnsembleTrait};
use orion::numbers::NumberTrait;

use alexandria_data_structures::merkle_tree::{pedersen::PedersenHasherImpl};
use alexandria_data_structures::array_ext::{SpanTraitExt};

#[derive(Destruct)]
struct TreeEnsembleClassifier<T> {
    ensemble: TreeEnsemble<T>,
    class_ids: Span<usize>,
    class_nodeids: Span<usize>,
    class_treeids: Span<usize>,
    class_weights: Span<T>,
    class_labels: Span<usize>,
    base_values: Option<Span<T>>,
    post_transform: PostTransform,
}

#[derive(Copy, Drop)]
enum PostTransform {
    None,
    Softmax,
    Logistic,
    SoftmaxZero,
    Probit,
}


#[generate_trait]
impl TreeEnsembleClassifierImpl<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +TensorTrait<usize>,
    +TensorTrait<T>,
> of TreeEnsembleClassifierTrait<T> {
    fn predict(ref self: TreeEnsembleClassifier<T>, X: Tensor<T>) -> (Tensor<usize>, Tensor<T>) {
        let leaf_indices = self.ensemble.leave_index_tree(X);
        let scores = compute_scores(ref self, leaf_indices);
        let (predictions, final_scores) = classify(ref self, scores);

        (predictions, final_scores)
    }
}


fn compute_scores<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>,>(
    ref self: TreeEnsembleClassifier<T>, leaf_indices: Tensor<usize>
) -> (Span<usize>, Felt252Dict::<Nullable<T>>) {
    // Initialize the scores array, either with base_values or zeros
    let num_samples = *leaf_indices.shape[0];
    let num_classes = self.class_labels.len();
    let mut scores_shape = array![num_samples, num_classes].span();

    // Store scores in dictionary because of immutability of array.
    let mut scores_data: Felt252Dict<Nullable<T>> = Default::default();
    if self.base_values.is_some() {
        // Repeat base values for each sample
        let mut sample_index: usize = 0;
        loop {
            if sample_index == num_samples {
                break;
            }

            let mut class_index: usize = 0;
            loop {
                if class_index == num_classes {
                    break;
                }

                let mut key = PedersenHasherImpl::new();
                let key: felt252 = key.hash(sample_index.into(), class_index.into());
                scores_data
                    .insert(key, NullableTrait::new(*self.base_values.unwrap().at(class_index)));

                class_index += 1;
            };

            sample_index += 1;
        }
    }

    // Compute class index mapping
    let mut class_index: Felt252Dict<Nullable<Span<(usize, T)>>> = Default::default();
    let mut class_weights = self.class_weights;
    let mut class_ids = self.class_ids;
    let mut class_nodeids = self.class_nodeids;
    let mut class_treeids = self.class_treeids;
    loop {
        match class_weights.pop_front() {
            Option::Some(class_weight) => {
                let mut class_id: usize = *class_ids.pop_front().unwrap();
                let mut node_id: usize = *class_nodeids.pop_front().unwrap();
                let mut tree_id: usize = *class_treeids.pop_front().unwrap();

                let mut key = PedersenHasherImpl::new();
                let key: felt252 = key.hash(tree_id.into(), node_id.into());

                let prev_value = class_index.get(key);
                match match_nullable(prev_value) {
                    FromNullableResult::Null(()) => { // entry.finalize(NullableTrait::new(array![]))
                    },
                    FromNullableResult::NotNull(val) => {
                        let mut new_val = prev_value.deref();
                        let new_va = new_val.concat(array![(class_id, *class_weight)].span());
                        class_index.insert(key, nullable_from_box(BoxTrait::new(new_va)));
                    },
                };
            },
            Option::None(_) => { break; }
        };
    };

    // Update scores based on class index mapping
    let mut sample_index: usize = 0;
    let mut leaf_indices_data = leaf_indices.data;
    loop {
        match leaf_indices_data.pop_front() {
            Option::Some(leaf_index) => {
                let mut key = PedersenHasherImpl::new();
                let key: felt252 = key
                    .hash(
                        (*self.ensemble.atts.nodes_treeids[*leaf_index]).into(),
                        (*self.ensemble.atts.nodes_nodeids[*leaf_index]).into()
                    );

                match match_nullable(class_index.get(key)) {
                    FromNullableResult::Null(()) => { continue; },
                    FromNullableResult::NotNull(class_weight_pairs) => {
                        let mut class_weight_pairs_span = class_weight_pairs.unbox();
                        loop {
                            match class_weight_pairs_span.pop_front() {
                                Option::Some(class_weight_pair) => {
                                    let (class_id, class_weight) = *class_weight_pair;

                                    let mut key = PedersenHasherImpl::new();
                                    let key: felt252 = key
                                        .hash((sample_index).into(), (class_id).into());

                                    let value = scores_data.get(key).deref();
                                    scores_data
                                        .insert(
                                            key,
                                            nullable_from_box(BoxTrait::new(value + class_weight))
                                        );
                                },
                                Option::None(_) => { break; }
                            };
                        }
                    },
                }

                sample_index += 1;
            },
            Option::None(_) => { break; }
        };
    };

    // Apply post-transform to scores
    match self.post_transform {
        PostTransform::None => {}, // No action required
        PostTransform::Softmax => panic_with_felt252(''),
        PostTransform::Logistic => panic_with_felt252(''),
        PostTransform::SoftmaxZero => panic_with_felt252(''),
        PostTransform::Probit => panic_with_felt252(''),
    }

    (scores_shape, scores_data)
}

fn classify<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +TensorTrait<usize>,
    +TensorTrait<T>,
>(
    ref self: TreeEnsembleClassifier<T>, scores: (Span<usize>, Felt252Dict::<Nullable<T>>)
) -> (Tensor<usize>, Tensor<T>) {
    let (scores_shape, mut scores_data) = scores;
    let num_samples = *scores_shape[0];
    let num_classes = *scores_shape[1];

    let predictions_shape = array![num_samples].span();
    let mut final_scores_shape = scores_shape;
    let mut predictions_data = ArrayTrait::new();
    let mut final_scores_data = ArrayTrait::new();

    let mut sample_index: usize = 0;
    loop {
        if sample_index == num_samples {
            break;
        }

        // Placeholder for the minimum value for type T
        let mut max_score = NumberTrait::<T>::min_value();
        let mut max_class_index = 0;

        let mut class_index: usize = 0;
        loop {
            if class_index == num_classes {
                break;
            }

            let mut key = PedersenHasherImpl::new();
            let key: felt252 = key.hash((sample_index).into(), (class_index).into());
            let score = scores_data[key].deref();

            if score > max_score {
                max_score = score;
                max_class_index = class_index;
            }

            class_index += 1;
        };

        final_scores_data.append(max_score);
        predictions_data.append(max_class_index);
        sample_index += 1;
    };

    (
        TensorTrait::new(predictions_shape, predictions_data.span()),
        TensorTrait::new(final_scores_shape, final_scores_data.span())
    )
}

