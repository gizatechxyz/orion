use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorDebugTrait};
use debug::PrintTrait;

fn print_shape<T, impl TTensorDebug: TensorDebugTrait<T>, impl TPrint: PrintTrait<T>>(self: Tensor<T>) {
    let mut i: usize = 0;
    loop {
        if i == (self.shape).len() {
            break ();
        };
        // '['.print();
        let shape_idx = self.shape.at(i);
        shape_idx.print();
        // ']'.print();
        i += 1;
    };
}
