use core::traits::Into;
use core::traits::TryInto;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use core::option::OptionTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;

use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};


/// Cf: NNTrait::space_to_depth docstring
fn space_to_depth<
    T,
    impl TTensor: TensorTrait<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    tensor: Tensor<T>, blocksize: usize
) -> Tensor<T> {
    assert!((tensor.shape).len() == 4, "Unexpected shape 4.");
    let b = (tensor.shape).at(0);
    let C = (tensor.shape).at(1);
    let H = (tensor.shape).at(2);
    let W = (tensor.shape).at(3);
    let tmpshape = array![*b, *C, *H / blocksize, blocksize, *W / blocksize, blocksize];
    let reshaped = (tensor).reshape(target_shape: tmpshape.span());
    let transposed = reshaped.transpose(axes: array![0, 3, 5, 1, 2, 4].span());
    let finalshape = array![*b, *C * blocksize * blocksize, *H / blocksize, *W / blocksize];
    return transposed.reshape(target_shape: finalshape.span());
}
