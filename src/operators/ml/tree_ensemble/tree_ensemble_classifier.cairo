use orion::numbers::NumberTrait;
use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{TreeEnsemble, TreeEnsembleTrait, TreeEnsembleImpl};
use orion::utils::get_row;

use alexandria_data_structures::merkle_tree::{pedersen::PedersenHasherImpl};
use alexandria_data_structures::array_ext::ArrayTraitExt;
use alexandria_data_structures::vec::{VecTrait, NullableVec};

#[derive(Copy, Drop)]
enum PostTransform {
    None,
    Logistic,
    Softmax,
    SoftmaxZero,
    Probit,
}

fn classify<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>, +PartialEq<T>, +Add<T>,
// +Drop<(Felt252DictEntry::<Nullable<T>>, Nullable::<T>)>,
// +Copy<Felt252Dict<Nullable<T>>>
>(
    ref self: TreeEnsemble<T>,
    X: Tensor<T>,
    post_transform: PostTransform,
    class_labels: Span<usize>
) {
    let leaves_index = self.leave_index_tree(X);
    let n_classes = class_labels.len();
    let res_shape: Span<usize> = array![*leaves_index.shape[0], n_classes].span();
}

