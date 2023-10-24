use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorDebugTrait};
use debug::PrintTrait;

fn print_data<T, impl TTensorDebug: TensorDebugTrait<T>, impl TPrint: PrintTrait<T>>(self: Tensor<T>) {
    let mut i: usize = 0;
    loop {
        if i == (self.data).len() {
            break ();
        };
        let data_idx = self.data.at(i);
        data_idx.print();
        ', '.print();
        i += 1;
    };
}
