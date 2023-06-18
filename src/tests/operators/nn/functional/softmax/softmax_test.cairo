use array::SpanTrait;
use traits::Into;

use orion::tests::operators::tensor::helpers::helpers_i32::i32_tensor_2x2_helper;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32;

use debug::print_felt252;

#[test]
#[available_gas(20000000)]
fn softmax_test() {
    let tensor = i32_tensor_2x2_helper();
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
    let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

    let mut result = NNTrait::softmax(@tensor, 0).data;

    assert((*result.at(0)).into() == 999946, 'result[0] = 0.1192');
    assert((*result.at(1)).into() == 999946, 'result[1] = 0.1192');
    assert((*result.at(2)).into() == 7388661, 'result[2] = 0.8808');
    assert((*result.at(3)).into() == 7388661, 'result[3] = 0.8808');

    let mut result = NNTrait::softmax(@tensor, 1).data;

    assert((*result.at(0)).into() == 2256044, 'result[0] = 0.2689');
    assert((*result.at(1)).into() == 6132563, 'result[1] = 0.7311');
    assert((*result.at(2)).into() == 2256043, 'result[2] = 0.2689');
    assert((*result.at(3)).into() == 6132564, 'result[4] = 0.7311');
}
