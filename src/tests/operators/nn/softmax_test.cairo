use array::SpanTrait;

use onnx_cairo::tests::operators::tensor::helpers::i32_tensor_2x2_helper;
use onnx_cairo::operators::nn::nn_i32::NN;

use debug::print_felt252;

#[test]
#[available_gas(20000000)]
fn softmax_test() {
    let tensor = i32_tensor_2x2_helper();

    let mut result = NN::softmax(@tensor, 0).data;

    assert(*result.at(0).mag == 7999572, 'result[0] = 0.1192');
    assert(*result.at(1).mag == 7999572, 'result[1] = 0.1192');
    assert(*result.at(2).mag == 59109291, 'result[2] = 0.8808');
    assert(*result.at(3).mag == 59109291, 'result[3] = 0.8808');

    let mut result = NN::softmax(@tensor, 1).data;

    assert(*result.at(0).mag == 18048353, 'result[0] = 0.2689');
    assert(*result.at(1).mag == 49060510, 'result[1] = 0.7311');
    assert(*result.at(2).mag == 18048352, 'result[2] = 0.2689');
    assert(*result.at(3).mag == 49060511, 'result[4] = 0.7311');
}
