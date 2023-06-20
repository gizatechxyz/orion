use array::SpanTrait;

use traits::Into;

use orion::tests::operators::tensor::helpers::helpers_i32::i32_tensor_2x2_helper;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use debug::PrintTrait;

use debug::print_felt252;

#[test]
#[available_gas(20000000)]
fn logsoftmax_test() {
    let tensor = i32_tensor_2x2_helper();
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
    let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

    let mut result = NNTrait::logsoftmax(@tensor, 0).data;

    assert(*result.at(0).mag == 17841970, 'result[0] = -2.12693');
    assert(*result.at(1).mag == 17841970, 'result[1] = -2.12695');
    assert(*result.at(2).mag == 1064751, 'result[2] = -0.12692');
    assert(*result.at(3).mag == 1064751, 'result[3] = -0.12692');

     let mut result = NNTrait::logsoftmax(@tensor, 1).data;

    assert(*result.at(0).mag == 11016451, 'result[0] = -1.3134');
    assert(*result.at(1).mag == 2627827, 'result[1] = -0.3132');
    assert(*result.at(2).mag == 11016460, 'result[2] = -1.3134');
    assert(*result.at(3).mag == 2627829, 'result[3] = -0.3132');
}
