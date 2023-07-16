use array::SpanTrait;
use array::ArrayTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32, u32TensorAdd};
use orion::performance::core::PerfomanceTrait;

/// Cf: NNTrait::convint docstring
fn convint_u32(z: Tensor<u32>, weights: Tensor<u32>, bias: Tensor<u32>, kernelSize: usize, strides: usize) -> Tensor<u32> {
    assert(z.shape.len() < 3, 'input tensor be at least 3D');

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    let extra = Option::<ExtraParams>::None(());
    let output = TensorTrait::new(shape.span(), data.span(), extra);

    return output;
}
