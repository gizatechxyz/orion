use core::array::ArrayTrait;


#[derive(Copy, Drop)]
pub struct Tree {
    pub base_weights: Span<i32>,
    pub left_children: Span<u32>,
    pub right_children: Span<u32>,
    pub is_left_leaf: Span<bool>,
    pub is_right_leaf: Span<bool>,
    pub split_indices: Span<u32>,
    pub split_conditions: Span<i32>
}

pub fn accumulate_scores_from_trees(mut trees: Span<Tree>, features: Span<i32>) -> i32 {
    let mut accumulated_score = 0;

    loop {
        match trees.pop_front() {
            Option::Some(tree) => {
                let mut score_from_tree = navigate_tree_and_accumulate_score(
                    *tree, features, 0, false
                );
                accumulated_score += score_from_tree;
            },
            Option::None => { break; },
        }
    };

    accumulated_score
}

fn navigate_tree_and_accumulate_score(
    tree: Tree, features: Span<i32>, mut node: u32, mut is_leaf: bool
) -> i32 {
    while !is_leaf {
        let feature_index = *tree.split_indices[node];
        let threshold = *tree.split_conditions[node];
        if *features.at(feature_index) < threshold {
            is_leaf = *tree.is_left_leaf[node];
            node = *tree.left_children[node];
        } else {
            is_leaf = *tree.is_right_leaf[node];
            node = *tree.right_children[node];
        }
    };
    *tree.base_weights[node]
}
