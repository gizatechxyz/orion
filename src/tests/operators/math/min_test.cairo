use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::TensorTrait;
use orion::tests::operators::tensor::helpers::helpers_i32::i32_tensor_2x2x2_helper;

#[test]
#[available_gas(2000000)]
fn min_tensor() {
    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.min().mag;
    assert(result == 0_u32, 'tensor.min = 0');
}
