use core::array::SpanTrait;
use core::dict::Felt252DictTrait;
use core::Felt252Dict;

impl UsizeDictCopy of Copy<Felt252Dict<usize>>;
impl UsizeDictDrop of Drop<Felt252Dict<usize>>;

#[derive(Copy, Drop)]
struct TreeEnsembleAttributes<T> {
    nodes_modes: Span<NODE_MODES>, // Change to modes
    nodes_featureids: Span<usize>,
    nodes_missing_value_tracks_true: Span<usize>,
    nodes_values: Span<T>,
    nodes_truenodeids: Span<usize>,
    nodes_falsenodeids: Span<usize>,
}

#[derive(Copy, Drop)]
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


fn leaf_index_tree<T, +Drop<T>, +Copy<T>>(
    ref ensemble: TreeEnsemble<T>, x: Span<T>, tree_id: usize
) -> usize {
    let mut index = ensemble.root_index.get(tree_id.into());

    loop {
        match *ensemble.atts.nodes_modes.at(index) {
            NODE_MODES::BRANCH_LEQ => { continue; },
            NODE_MODES::BRANCH_LT => { continue; },
            NODE_MODES::BRANCH_GTE => { continue; },
            NODE_MODES::BRANCH_GT => { continue; },
            NODE_MODES::BRANCH_EQ => { continue; },
            NODE_MODES::BRANCH_NEQ => { continue; },
            NODE_MODES::LEAF => { break; },
        };
    };

    1
}

