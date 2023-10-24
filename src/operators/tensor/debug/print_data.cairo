use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorDebugTrait, TensorTrait};
use debug::PrintTrait;

fn print_data<
    T, 
    impl TTensorDebug: TensorDebugTrait<T>,
    impl TTensor: TensorTrait<T>,
    impl TPrint: PrintTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
    > (self: Tensor<T>) {
    let mut i: usize = 0;
    loop {
        if i == (self.data).len() {
            break ();
        };
        let data_idx = self.data.at(i);
        (*data_idx).print();
        ', '.print();
        i += 1;
    };
}
