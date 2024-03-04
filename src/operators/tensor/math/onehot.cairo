use orion::numbers::NumberTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::onehot docstring
fn onehot_encode<
    T,
    MAG,
    impl FFixed: FixedTrait<T, MAG>,
    impl FTensorTrait: TensorTrait<T>,
    impl FNumber: NumberTrait<T, MAG>,
    impl U32TryIntoMAG: TryInto<u32, MAG>,
    impl FPartialEq: PartialEq<T>,
    impl FAdd: Add<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    self: @Tensor<T>, depth: usize, axis: Option<usize>, values: Tensor<T>
) -> Tensor<T> {
    let mut data = *self.data;
    let mut shape = *self.shape;
    let rank = shape.len();

    // using 999 to denote -1, innermost dimension
    let axis = match axis {
        Option::Some(val) => val,
        Option::None => 999
    };

    assert(((axis == 999) | (axis.into() <= rank)), 'axis out of dimensions');

    let mut output_data = array![];
    let mut output_size: Array<usize> = array![];

    // New shape for output data
    loop {
        match shape.pop_front() {
            Option::Some(size) => { output_size.append(*size); },
            Option::None => { break; }
        };
    };
    output_size.append(depth.into());

    // OneHot enocde loop
    loop {
        match data.pop_front() {
            Option::Some(outer_index) => {
                let mut fixed_number = *outer_index;

                if fixed_number.is_neg() {
                    fixed_number =
                        FixedTrait::<T, MAG>::new_unscaled(depth.try_into().unwrap(), false)
                        + fixed_number
                }

                let mut inner_index = 0;
                while inner_index != depth {
                    let ind = FixedTrait::<
                        T, MAG
                    >::new_unscaled(inner_index.try_into().unwrap(), false);

                    if fixed_number == ind {
                        output_data.append(*values.data.at(1));
                    } else {
                        output_data.append(*values.data.at(0));
                    };

                    inner_index += 1;
                };
            },
            Option::None => { break; }
        };
    };

    let mut output_tensor = TensorTrait::new(output_size.span(), output_data.span());
    let mut tranpose_axes = array![];
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

        output_tensor = output_tensor.transpose(tranpose_axes.span());
    }

    output_tensor
}

fn onehot<
    T,
    MAG,
    impl FFixed: FixedTrait<T, MAG>,
    impl FTensorTrait: TensorTrait<T>,
    impl FNumber: NumberTrait<T, MAG>,
    impl U32TryIntoMAG: TryInto<u32, MAG>,
    impl FPartialEq: PartialEq<T>,
    impl FAdd: Add<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    self: @Tensor<T>, depth: usize, axis: Option<usize>, mut values: Span<usize>,
) -> Tensor<T> {
    assert(values.len() == 2, 'Wrong values dimensions');

    let mut sizes = array![];
    sizes.append(2);

    let mut first = *values.pop_front().unwrap();
    let mut second = *values.pop_front().unwrap();

    let mut data = array![];
    data.append(FixedTrait::<T, MAG>::new_unscaled(first.try_into().unwrap(), false));
    data.append(FixedTrait::<T, MAG>::new_unscaled(second.try_into().unwrap(), false));

    let values = TensorTrait::new(sizes.span(), data.span());

    onehot_encode(self, depth, axis, values)
}
