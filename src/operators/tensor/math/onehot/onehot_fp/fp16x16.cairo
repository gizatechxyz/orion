use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use core::clone::Clone;
use option::OptionTrait;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16Add, FP16x16Sub, FP16x16AddEq, FP16x16PartialEq
};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use array::{ArrayTrait, SpanTrait};
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::{impl_tensor_i32, impl_tensor_u32};
use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;


/// Cf: TensorTrait::onehot docstring
fn onehot_encode(
    self: @Tensor<FixedType>, depth: usize, axis: Option<usize>, values: Tensor<FixedType>
) -> Tensor<FixedType> {
    let data = *self.data;
    let shape = *self.shape;
    let rank = shape.len();

    // using 999 to denote -1, innermost dimension
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 999
    };

    assert(((axis == 999) | (axis <= rank)), 'axis out of dimensions');

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
    output_size.append(depth);

    // OneHot enocde loop
    let mut outer_index: usize = 0;
    loop {
        if outer_index == tensor_len {
            break ();
        };

        let mut inner_index: usize = 0;

        let mut fixed_number = *tensor_data.at(outer_index);
        if fixed_number.sign == true {
            fixed_number =
                FP16x16Add::add(FixedTrait::new_unscaled(depth.into(), false), fixed_number)
        }

        loop {
            if inner_index == depth {
                break ();
            };
            let ind = FixedTrait::new_unscaled(inner_index.into(), false);

            if fixed_number == ind {
                output_data.append(*values.data.at(1));
            } else {
                output_data.append(*values.data.at(0));
            };

            inner_index += 1;
        };

        outer_index += 1;
    };

    let mut output_tensor = TensorTrait::<FixedType>::new(
        output_size.span(), output_data.span(), extra
    );
    let mut tranpose_axes = ArrayTrait::<u32>::new();
    // Get New shape is axis is not last dimension
    if (axis != 999) & (axis != rank) {
        let mut index: usize = 0;
        loop {
            let max_dim = output_size.len() - 1;
            if index == max_dim {
                break ();
            };

            if axis == index {
                tranpose_axes.append(max_dim)
            }
            tranpose_axes.append(index);
            index += 1;
        };

        let mut index: usize = 0;

        output_tensor = output_tensor.transpose(tranpose_axes.span());
    }

    return output_tensor;
}

/// Cf: TensorTrait::onehot docstring
fn onehot(
    self: @Tensor<FixedType>, depth: usize, axis: Option<usize>, mut values: Span<usize>, 
) -> Tensor<FixedType> {
    assert(values.len() == 2, 'Wrong values dimensions');

    let mut sizes = ArrayTrait::new();
    sizes.append(2);

    let mut first = *values.pop_front().unwrap();
    let mut second = *values.pop_front().unwrap();

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(first.into(), false));
    data.append(FixedTrait::new_unscaled(second.into(), false));
    let extra = Option::<ExtraParams>::None(());

    let values = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    onehot_encode(self, depth, axis, values)
}
