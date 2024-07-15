use alexandria_data_structures::span_ext::SpanTraitExt;
use alexandria_merkle_tree::merkle_tree::{pedersen::PedersenHasherImpl};
use alexandria_data_structures::array_ext::ArrayTraitExt;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use orion::utils::get_row;

#[derive(Copy, Drop, Destruct)]
struct TreeEnsembleAttributes<T> {
    nodes_falsenodeids: Span<usize>,
    nodes_featureids: Span<usize>,
    nodes_missing_value_tracks_true: Span<usize>,
    nodes_modes: Span<NODE_MODES>,
    nodes_nodeids: Span<usize>,
    nodes_treeids: Span<usize>,
    nodes_truenodeids: Span<usize>,
    nodes_values: Span<T>,
}

#[derive(Destruct)]
struct TreeEnsemble<T> {
    atts: TreeEnsembleAttributes<T>,
    tree_ids: Span<usize>,
    root_index: Felt252Dict<usize>,
    node_index: Felt252Dict<usize>, // index is pedersen hash of tree_id and nid.
}

#[derive(Copy, Drop)]
enum NODE_MODES {
    BRANCH_LEQ,
    BRANCH_LT,
    BRANCH_GTE,
    BRANCH_GT,
    BRANCH_EQ,
    BRANCH_NEQ,
    LEAF
}

#[generate_trait]
impl TreeEnsembleImpl<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>, +PartialEq<T>
> of TreeEnsembleTrait<T> {
    fn leaf_index_tree(ref self: TreeEnsemble<T>, x: Span<T>, tree_id: usize) -> usize {
        let mut index: usize = self.root_index.get(tree_id.into());

        loop {
            // Loop breaker
            match *self.atts.nodes_modes.at(index) {
                NODE_MODES::BRANCH_LEQ => {},
                NODE_MODES::BRANCH_LT => {},
                NODE_MODES::BRANCH_GTE => {},
                NODE_MODES::BRANCH_GT => {},
                NODE_MODES::BRANCH_EQ => {},
                NODE_MODES::BRANCH_NEQ => {},
                NODE_MODES::LEAF => { break; },
            };

            let x_value = *x.at(*(self.atts.nodes_featureids).at(index));
            let r = if x_value.is_nan() {
                *self.atts.nodes_missing_value_tracks_true.at(index) >= 1
            } else {
                match *self.atts.nodes_modes.at(index) {
                    NODE_MODES::BRANCH_LEQ => x_value <= *self.atts.nodes_values[index],
                    NODE_MODES::BRANCH_LT => x_value < *self.atts.nodes_values[index],
                    NODE_MODES::BRANCH_GTE => x_value >= *self.atts.nodes_values[index],
                    NODE_MODES::BRANCH_GT => x_value > *self.atts.nodes_values[index],
                    NODE_MODES::BRANCH_EQ => x_value == *self.atts.nodes_values[index],
                    NODE_MODES::BRANCH_NEQ => x_value != *self.atts.nodes_values[index],
                    NODE_MODES::LEAF => {
                        panic(array!['Unexpected rule for node index ', index.into()])
                    },
                }
            };

            let nid = if r {
                *self.atts.nodes_truenodeids[index]
            } else {
                *self.atts.nodes_falsenodeids[index]
            };

            // key of TreeEnsemble.node_index is pedersen hash of tree_id and nid.
            let mut key = PedersenHasherImpl::new();
            let key: felt252 = key.hash(tree_id.into(), nid.into());

            index = self.node_index.get(key);
        };

        index
    }

    fn leave_index_tree(ref self: TreeEnsemble<T>, x: Tensor<T>) -> Tensor<usize> {
        let mut outputs: Array<usize> = array![];

        let mut i: usize = 0;
        let breaker: usize = *x.shape[0];
        while i != breaker {
            let row_data: Span<T> = get_row(@x, i);
            let mut outs: Array<usize> = array![];
            let mut tree_ids = self.tree_ids;
            loop {
                match tree_ids.pop_front() {
                    Option::Some(tree_id) => {
                        outs
                            .append(
                                TreeEnsembleImpl::<T>::leaf_index_tree(ref self, row_data, *tree_id)
                            )
                    },
                    Option::None => { break; }
                };
            };

            outputs.append_all(ref outs);
            i += 1;
        };

        TensorTrait::new(array![*x.shape[0], self.tree_ids.len()].span(), outputs.span())
    }
}

