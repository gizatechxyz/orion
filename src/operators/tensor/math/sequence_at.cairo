use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use orion::numbers::signed_integer::i32::i32;

/// Cf: TensorTrait::sequence_at docstring
fn sequence_at<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    sequence: Array<Tensor<T>>, index: Tensor<i32>
) -> Tensor<T> {

    assert(index.shape.len() == 0 && index.data.len() == 1, 'Index must be a scalar');

    let index_value_i32: i32 = *index.data.at(0);
    let is_negative: bool = index_value_i32.sign;
    let index_value = index_value_i32.mag;

    assert((is_negative == false && index_value <= sequence.len() - 1) || (is_negative == true && index_value <= sequence.len()), 'Index out of bounds');
    
    if is_negative == false {
        return *sequence.at(index_value);
    } else {
        let reverted_index_value = sequence.len() - index_value;
        return *sequence.at(reverted_index_value);
    }
}