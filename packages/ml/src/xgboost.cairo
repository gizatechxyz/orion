use core::array::{ArrayTrait, SpanTrait};

#[derive(Copy, Drop)]
pub struct Tree {
    pub base_weights: Span<i32>,
    pub left_children: Span<u32>,
    pub right_children: Span<u32>,
    pub split_indices: Span<u32>,
    pub split_conditions: Span<i32>
}

pub fn accumulate_scores_from_trees(mut trees: Span<Tree>, features: Span<i32>) -> i32 {
    let mut accumulated_score = 0;

    loop {
        match trees.pop_front() {
            Option::Some(tree) => {
                let score_from_tree = navigate_tree_and_accumulate_score(*tree, features, 0);
                accumulated_score += score_from_tree;
            },
            Option::None => { break; },
        }
    };

    accumulated_score
}

fn navigate_tree_and_accumulate_score(tree: Tree, features: Span<i32>, mut node: u32) -> i32 {
    loop {
        if *tree.left_children[node] == 0 {
            if *tree.right_children[node] == 0 {
                break *tree.base_weights[node];
            }
        }

        let feature_index = *tree.split_indices[node];
        let threshold = *tree.split_conditions[node];
        if *features.at(feature_index) < threshold {
            node = *tree.left_children[node];
        } else {
            node = *tree.right_children[node];
        }
    }
}
