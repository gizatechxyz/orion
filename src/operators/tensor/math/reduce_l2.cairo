use core::option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;
use debug::PrintTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::fixed_point::core::FixedTrait;

fn square<
    T,
    MAG,
    impl FTensorTrait: TensorTrait<T>,
    impl FFixed: FixedTrait<T, MAG>,
    impl FNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    self: @Tensor<T>
) -> Tensor<T> {
    let mut data = *self.data;
    let mut output_data = ArrayTrait::new();

    loop {
        match data.pop_front() {
            Option::Some(item) => { 
                let ele = *item;
                output_data.append(ele * ele); 
                },
            Option::None(_) => { break; }
        };
    };

    let tensor_square = TensorTrait::new(*self.shape, output_data.span());
    return tensor_square;
}
/// Cf: TensorTrait::reduce_l2 docstring
fn reduce_l2<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl FFixed: FixedTrait<T, MAG>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, keepdims: bool
) -> Tensor<T> {
    let tensor_square = square(self);
    let tensor_square_sum =  tensor_square.reduce_sum(axis: axis, keepdims: keepdims);
    return tensor_square_sum.sqrt();

}



// Tests --------------------------------------------------------------------------------------------------------------

use orion::numbers::fixed_point::implementations::fp8x23::helpers::assert_precise;

use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::assert_eq;

use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;

fn data() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(5, false));

    let tensor = TensorTrait::<FP8x23>::new(shape.span(), data.span());

    return tensor;
}

#[test]
#[available_gas(2000000000)]
fn test_reduce_l2_default() {
    let mut data = data();

    let y = data.reduce_l2(axis: 1,  keepdims: true);
    let mut output = y.data;

    loop {
        match output.pop_front() {
            Option::Some(item) => { 
                (*item).print();
                },
            Option::None(_) => { break; }
        };
    };
    
}
