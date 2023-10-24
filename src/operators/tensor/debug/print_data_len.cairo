use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorDebugTrait};
use debug::PrintTrait;

fn print_data_len<T, impl TTensorDebug: TensorDebugTrait<T>, impl TPrint: PrintTrait<T>>(self: Tensor<T>) -> usize {
    let data_len = self.data.len();
    data_len.print();
    data_len
}
