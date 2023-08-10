use core::traits::Into;
use debug::PrintTrait;
use core::traits::TryInto;
use core::serde::Serde;
use core::traits::Destruct;
use core::clone::Clone;
use option::OptionTrait;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use array::{ArrayTrait, SpanTrait};


/// Cf: TensorTrait::onehot docstring
fn onehot_encode(
    self: @Tensor<u32>, depth: usize, axis: Option<usize>, values: Tensor<u32>
) -> Tensor<u32> {
    let data = *self.data;
    let shape = *self.shape;
    let rank = shape.len();

    // using 999 to denote -1, innermost dimension
    let axis = match axis {
        Option::Some(val) => val,
        Option::None(_) => 999
    };

    assert(((axis == 999) | (axis.into() <= rank)), 'axis out of dimensions');

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

        let mut inner_index: usize = 0;
        let mut fixed_number = *tensor_data.at(outer_index);

        loop {
            if inner_index == depth {
                break ();
            };
            let ind = inner_index;

            if fixed_number == ind {
                output_data.append(*values.data.at(1));
            } else {
                output_data.append(*values.data.at(0));
            };

            inner_index += 1;
        };

        outer_index += 1;
    };

    let mut output_tensor = TensorTrait::<u32>::new(output_size.span(), output_data.span(), extra);
    let mut tranpose_axes = ArrayTrait::new();
    // Get New shape is axis is not last dimension
    if (axis != 999) & (axis.into() != rank) {
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
    self: @Tensor<u32>, depth: usize, axis: Option<usize>, mut values: Span<usize>, 
) -> Tensor<u32> {
    assert(values.len() == 2, 'Wrong values dimensions');

    let mut sizes = ArrayTrait::new();
    sizes.append(2);

    let mut first: u32 = *values.pop_front().unwrap();
    let mut second: u32 = *values.pop_front().unwrap();

    let mut data = ArrayTrait::new();
    data.append(first);
    data.append(second);

    let extra = Option::<ExtraParams>::None(());

    let values = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

    onehot_encode(self, depth, axis, values)
}
