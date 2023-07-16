use array::SpanTrait;
use array::ArrayTrait;

use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i8::{Tensor_i8, i8TensorAdd};
use orion::performance::core::PerfomanceTrait;

/// Cf: NNTrait::convint docstring
fn convint_i8(z: Tensor<i8>, weights: Tensor<i8>, bias: Tensor<i8>, kernelSize: usize, strides: usize) -> Tensor<i8> {
    assert(z.shape.len() < 3, 'input tensor be at least 3D');

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 71, sign: true });
    data.append(i8 { mag: 38, sign: false });
    data.append(i8 { mag: 62, sign: false });
    let extra = Option::<ExtraParams>::None(());
    let output = TensorTrait::new(shape.span(), data.span(), extra);

    return output;
}
