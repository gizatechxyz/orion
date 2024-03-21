use core::option::OptionTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
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
    assert((tensor.shape).len() == 4, 'Unexpected shape 4.');

    let blocksize_i32: i32 = blocksize.try_into().unwrap();

    let b: i32 = (*(tensor.shape).at(0)).try_into().unwrap();
    let C: i32 = (*(tensor.shape).at(1)).try_into().unwrap();
    let H: u32 = (*(tensor.shape).at(2));
    let W: u32 = (*(tensor.shape).at(3));
    let tmpshape = array![
        b,
        C,
        (H / blocksize).try_into().unwrap(),
        blocksize_i32,
        (W / blocksize).try_into().unwrap(),
        blocksize_i32
    ];
    let reshaped = (tensor).reshape(target_shape: tmpshape.span());
    let transposed = reshaped.transpose(axes: array![0, 3, 5, 1, 2, 4].span());
    let finalshape = array![
        b,
        C * blocksize_i32 * blocksize_i32,
        (H / blocksize).try_into().unwrap(),
        (W / blocksize).try_into().unwrap()
    ];

    transposed.reshape(target_shape: finalshape.span())
}
