use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use core::clone::Clone;
use option::OptionTrait;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use array::{ArrayTrait, SpanTrait};
use orion::numbers::signed_integer::i8::{
    i8Add, i8AddEq, i8PartialEq, i8PartialOrd, i8_new, i8Into,
};


/// Cf: TensorTrait::onehot docstring
fn onehot_encode(
    self: @Tensor<i8>, depth: usize, axis: Option<usize>, values: Tensor<i8>
) -> Tensor<i8> {
    let data = *self.data;
    let shape = *self.shape;
    let rank = shape.len();

    // using 255 to denote -1, innermost dimension
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 255
    };

    assert(((axis == 255) | (axis.into() <= rank)), 'axis out of dimensions');

    let mut tensor_data = self.data.clone();
    let tensor_len: usize = data.len();

    let mut output_data = ArrayTrait::new();
    let mut output_size = ArrayTrait::<u32>::new();
    let extra = Option::<ExtraParams>::None(());

    // New shape for output data
    let mut index: usize = 0;
    loop {
        if index == shape.len() {
            break ();
        };
        let size: usize = *shape.at(index);
        output_size.append(size);
        index += 1;
    };
    output_size.append(depth.into());

    // OneHot enocde loop
    let mut outer_index: usize = 0;
    loop {
        if outer_index == tensor_len {
            break ();
        };

        let mut inner_index: u8 = 0;
        let mut fixed_number = *tensor_data.at(outer_index);
        let depth: u8 = integer::u8_try_from_felt252(integer::u32_to_felt252(depth)).unwrap();

        if fixed_number.sign == true {
            fixed_number = i8Add::add(i8_new(depth, false), fixed_number)
        }

        loop {
            if inner_index == depth {
                break ();
            };
            let ind = IntegerTrait::<i8>::new(inner_index, false);

            if fixed_number == ind {
                output_data.append(*values.data.at(1));
            } else {
                output_data.append(*values.data.at(0));
            };

            inner_index += 1;
        };

        outer_index += 1;
    };

    let mut output_tensor = TensorTrait::<i8>::new(output_size.span(), output_data.span(), extra);
    let mut tranpose_axes = ArrayTrait::new();
    // Get New shape is axis is not last dimension
    if (axis != 255) & (axis.into() != rank) {
        let mut index: usize = 0;
        loop {
            let max_dim = output_size.len() - 1;
            if index.into() == max_dim {
                break ();
            };

            if axis == index {
                tranpose_axes.append(max_dim.into())
            }
            tranpose_axes.append(index.into());
            index += 1;
        };

        let mut index: usize = 0;

        output_tensor = output_tensor.transpose(tranpose_axes.span());
    }

    return output_tensor;
}


fn onehot(
    self: @Tensor<i8>, depth: usize, axis: Option<usize>, mut values: Span<usize>, 
) -> Tensor<i8> {
    assert(values.len() == 2, 'Wrong values dimensions');

    let mut sizes = ArrayTrait::new();
    sizes.append(2);

    let mut first = *values.pop_front().unwrap();
    let mut second = *values.pop_front().unwrap();

    let first = integer::u32_to_felt252(first);
    let second = integer::u32_to_felt252(second);
    let first: u8 = integer::u8_try_from_felt252(first).unwrap();
    let second: u8 = integer::u8_try_from_felt252(second).unwrap();

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::<i8>::new(first, false));
    data.append(IntegerTrait::<i8>::new(second, false));
    let extra = Option::<ExtraParams>::None(());

    let values = TensorTrait::<i8>::new(sizes.span(), data.span(), extra);
    onehot_encode(self, depth, axis, values)
}
