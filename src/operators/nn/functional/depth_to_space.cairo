use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};

/// Cf: NNTrait::depth_to_space docstring
fn depth_to_space<
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
    tensor: Tensor<T>, blocksize: usize, mode: felt252
) -> Tensor<T> {
    assert((tensor.shape).len() == 4, 'Unexpected shape 4.');

    let b = (tensor.shape).at(0);
    let C = (tensor.shape).at(1);
    let H = (tensor.shape).at(2);
    let W = (tensor.shape).at(3);
    let finalshape = array![*b, *C / (blocksize * blocksize), *H * blocksize, *W * blocksize];

    if mode == 'DCR' {
        let tmpshape = array![*b, blocksize, blocksize, *C / (blocksize * blocksize), *H, *W];
        let reshaped = (tensor).reshape(target_shape: tmpshape.span(), allowzero: Option::None);
        let transposed = reshaped.transpose(axes: array![0, 3, 4, 1, 5, 2].span());

        transposed.reshape(target_shape: finalshape.span(), allowzero: Option::None)
    } else {
        // assert mode == "CRD"
        let tmpshape = array![*b, *C / (blocksize * blocksize), blocksize, blocksize, *H, *W];
        let reshaped = (tensor).reshape(target_shape: tmpshape.span(), allowzero: Option::None);
        let transposed = reshaped.transpose(axes: array![0, 1, 4, 2, 5, 3].span());

        transposed.reshape(target_shape: finalshape.span(), allowzero: Option::None)
    }
}
