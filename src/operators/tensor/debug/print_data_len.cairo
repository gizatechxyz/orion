use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, TensorDebugTrait};
use debug::PrintTrait;

fn print_data_len<
    T, 
    impl TTensorDebug: TensorDebugTrait<T>,
    impl TTensor: TensorTrait<T>,
    impl TPrint: PrintTrait<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
    > (self: Tensor<T>) -> usize {
    let data_len = self.data.len();
    data_len.print();
    data_len
}

// #[test]
// fn test_print_data_len() {
//     let arr_data = array![0; 27].span();
//     let tensor = TensorTrait::<i8>::new(shape: array![3, 3, 3].span(), data: arr_data);
//     let data_len = tensor.print_data_len();
//     assert_eq!(data_len, 27);
// }