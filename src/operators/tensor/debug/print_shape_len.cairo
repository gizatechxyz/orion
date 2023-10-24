use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorDebugTrait, TensorTrait};
use debug::PrintTrait;

fn print_shape_len<
    T, 
    impl TTensorDebug: TensorDebugTrait<T>,
    impl TTensor: TensorTrait<T>,
    impl TPrint: PrintTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
    > (self: Tensor<T>) -> usize {
    let shape_len = self.shape.len();
    shape_len.print();
    shape_len
}
