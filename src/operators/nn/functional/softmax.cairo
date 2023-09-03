use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: NNTrait::softmax docstring
fn softmax<
    T,
    T,
    impl TTensor: TensorTrait<T>,
    impl TTensor: TensorTrait<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    z: @Tensor<T>, axis: usize
) -> Tensor<T> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;

    return softmax;
}

