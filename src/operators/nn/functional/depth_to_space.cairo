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

    let blocksize_i32: i32 = blocksize.try_into().unwrap();

    let b: i32 = (*(tensor.shape).at(0)).try_into().unwrap();
    let C: u32 = (*(tensor.shape).at(1)).try_into().unwrap();
    let H: i32 = (*(tensor.shape).at(2)).try_into().unwrap();
    let W: i32 = (*(tensor.shape).at(3)).try_into().unwrap();
    let finalshape: Array<i32> = array![
        b,
        (C / (blocksize * blocksize)).try_into().unwrap(),
        (H * blocksize_i32),
        (W * blocksize_i32)
    ];

    if mode == 'DCR' {
        let tmpshape: Array<i32> = array![
            b, blocksize_i32, blocksize_i32, (C / (blocksize * blocksize)).try_into().unwrap(), H, W
        ];
        let reshaped = (tensor).reshape(target_shape: tmpshape.span());
        let transposed = reshaped.transpose(axes: array![0, 3, 4, 1, 5, 2].span());

        transposed.reshape(target_shape: finalshape.span())
    } else {
        // assert mode == "CRD"
        let tmpshape: Array<i32> = array![
            b, (C / (blocksize * blocksize)).try_into().unwrap(), blocksize_i32, blocksize_i32, H, W
        ];
        let reshaped = (tensor).reshape(target_shape: tmpshape.span());
        let transposed = reshaped.transpose(axes: array![0, 1, 4, 2, 5, 3].span());

        transposed.reshape(target_shape: finalshape.span())
    }
}
