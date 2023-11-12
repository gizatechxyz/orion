use core::array::ArrayTrait;
use core::clone::Clone;
use core::box::BoxTrait;
use core::traits::Into;
use core::option::OptionTrait;
use orion::operators::matrix::MutMatrixTrait;
use core::array::SpanTrait;
use core::nullable::NullableTrait;
use core::dict::Felt252DictTrait;
use core::dict::Felt252DictEntryTrait;
use nullable::{match_nullable, FromNullableResult};

use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::ml::tree_ensemble::core::{TreeEnsemble, TreeEnsembleImpl, TreeEnsembleTrait};
use orion::numbers::NumberTrait;
use orion::utils::get_row;

use alexandria_data_structures::merkle_tree::{pedersen::PedersenHasherImpl};
use alexandria_data_structures::array_ext::{SpanTraitExt};

use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};

use debug::PrintTrait;

#[derive(Destruct)]
struct TreeEnsembleClassifier<T> {
    ensemble: TreeEnsemble<T>,
    class_ids: Span<usize>,
    class_nodeids: Span<usize>,
    class_treeids: Span<usize>,
    class_weights: Span<T>,
    classlabels: Span<usize>,
    base_values: Option<Span<T>>,
    post_transform: POST_TRANSFORM,
}

#[derive(Copy, Drop)]
enum POST_TRANSFORM {
    NONE,
    SOFTMAX,
    LOGISTIC,
    SOFTMAXZERO,
    PROBIT,
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
    +PrintTrait<T>,
    +AddEq<T>,
    +Div<T>,
    +Mul<T>
> of TreeEnsembleClassifierTrait<T> {
    fn predict(ref self: TreeEnsembleClassifier<T>, X: Tensor<T>) -> (Span<usize>, MutMatrix::<T>) {
        let leaves_index = self.ensemble.leave_index_tree(X);
        let n_classes = self.classlabels.len();
        let mut res: MutMatrix<T> = MutMatrixImpl::new(*leaves_index.shape.at(0), n_classes);

        // Set base values
        if self.base_values.is_some() {
            let mut base_values = self.base_values.unwrap();
            let mut row: usize = 0;
            loop {
                if row == res.rows {
                    break;
                }

                let mut col: usize = 0;
                loop {
                    if col == res.cols {
                        break;
                    }

                    let value = *base_values.pop_front().unwrap();
                    res.set(row, col, value);

                    col += 1
                };

                row += 1;
            }
        } else {
            let mut row: usize = 0;
            loop {
                if row == res.rows {
                    break;
                }

                let mut col: usize = 0;
                loop {
                    if col == res.cols {
                        break;
                    }

                    res.set(row, col, NumberTrait::zero());

                    col += 1
                };

                row += 1;
            }
        }

        let mut class_index: Felt252Dict<Nullable<Span<usize>>> = Default::default();
        let mut i: usize = 0;
        loop {
            if i == self.class_treeids.len() {
                break;
            }

            let tid = *self.class_treeids[i];
            let nid = *self.class_nodeids[i];

            let mut key = PedersenHasherImpl::new();
            let key: felt252 = key.hash(tid.into(), nid.into());
            match match_nullable(class_index.get(key)) {
                FromNullableResult::Null(()) => {
                    class_index.insert(key, NullableTrait::new(array![i].span()));
                },
                FromNullableResult::NotNull(val) => {
                    let mut new_val = val.unbox();
                    let new_val = new_val.concat(array![i].span());
                    class_index.insert(key, NullableTrait::new(new_val));
                },
            }

            i += 1;
        };
        let mut i: usize = 0;
        loop {
            if i == res.rows {
                break;
            }

            let mut indices = get_row(@leaves_index, i);
            let mut t_index: Array<Span<core::integer::u32>> = ArrayTrait::new();
            loop {
                match indices.pop_front() {
                    Option::Some(index) => {
                        let mut key = PedersenHasherImpl::new();
                        let key: felt252 = key
                            .hash(
                                (*self.ensemble.atts.nodes_treeids[*index]).into(),
                                (*self.ensemble.atts.nodes_nodeids[*index]).into()
                            );
                        t_index.append(class_index.get(key).deref());
                    },
                    Option::None(_) => { break; }
                };
            };
            let mut t_index = t_index.span();
            loop {
                match t_index.pop_front() {
                    Option::Some(its) => {
                        let mut its = *its;
                        loop {
                            match its.pop_front() {
                                Option::Some(it) => {
                                    let prev_val = match res.get(i, *self.class_ids[*it]) {
                                        Option::Some(val) => {
                                            res
                                                .set(
                                                    i,
                                                    *self.class_ids[*it],
                                                    val + *self.class_weights[*it]
                                                );
                                        },
                                        Option::None => {
                                            res
                                                .set(
                                                    i,
                                                    *self.class_ids[*it],
                                                    *self.class_weights[*it]
                                                );
                                        },
                                    };
                                },
                                Option::None(_) => { break; }
                            };
                        };
                    },
                    Option::None(_) => { break; }
                };
            };
            i += 1;
        };

        // Post Transform
        let mut new_scores = match self.post_transform {
            POST_TRANSFORM::NONE => res, // No action required
            POST_TRANSFORM::SOFTMAX => res.softmax(1),
            POST_TRANSFORM::LOGISTIC => res.sigmoid(),
            POST_TRANSFORM::SOFTMAXZERO => panic_with_felt252('SoftmaxZero not supported yet'),
            POST_TRANSFORM::PROBIT => panic_with_felt252('Probit not supported yet'),
        };

        // Labels
        let mut labels = new_scores.argmax(1);

        let mut labels_list = ArrayTrait::new();
        loop {
            match labels.pop_front() {
                Option::Some(i) => { labels_list.append(*self.classlabels[*i]); },
                Option::None(_) => { break; }
            };
        };

        return (labels_list.span(), new_scores);
    }
}

