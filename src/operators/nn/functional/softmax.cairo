use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: NNTrait::softmax docstring
fn softmax<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl FTensor: TensorTrait<F, F>,
    impl FTensorDiv: Div<Tensor<F>>,
    impl FCopy: Copy<F>,
    impl FDrop: Drop<F>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<F> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}

