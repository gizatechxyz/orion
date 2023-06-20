use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::TensorTrait;
use orion::tests::operators::tensor::helpers::helpers_i32::i32_tensor_2x2x2_helper;

#[test]
#[available_gas(2000000)]
fn max_tensor() {
    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.max().mag;
    assert(result == 7_u32, 'tensor.max = 7');
}
