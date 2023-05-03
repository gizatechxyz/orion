use array::ArrayTrait;
use array::SpanTrait;
use traits::Into;

use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::tests::operators::tensor::helpers::i32_tensor_2x2_helper;
use onnx_cairo::operators::nn::nn_i32::nn;

use debug::print_felt252;

#[test]
#[available_gas(20000000)]
fn softmax_test() {
    let tensor = i32_tensor_2x2_helper();
    let mut result = nn::softmax(@tensor, 1).data;

    //assert((*result.at(0).mag).into() == 18048353, 'result[0] = 0.2689');
    

    print_felt252((*result.at(0).mag).into());
    print_felt252((*result.at(1).mag).into());
    print_felt252((*result.at(2).mag).into());
    print_felt252((*result.at(3).mag).into());
}
