use orion::numbers::NumberTrait;
use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};

/// Cf: TensorTrait::cumsum docstring
fn cumsum<
    T,
    MAG,
    impl TTensorTrait: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, exclusive: Option<bool>, reverse: Option<bool>
) -> Tensor<T> {
    let reverse = match reverse {
        Option::Some(val) => val,
        Option::None => false
    };

    if reverse {
        cumsum_reverse::<T>(self, axis, exclusive, NumberTrait::zero())
    } else {
        cumsum_forward::<T>(self, axis, exclusive, NumberTrait::zero())
    }
}

/// Cf: TensorTrait::cumsum docstring
fn cumsum_forward<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TAdd: Add<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, exclusive: Option<bool>, zero: T,
) -> Tensor<T> {
    let exclusive = match exclusive {
        Option::Some(val) => val,
        Option::None => false,
    };

    assert(axis < (*self.shape).len(), 'axis out of dimensions');

    let data = *self.data;

    let mut output_data = array![];

    let mut index: usize = 0;

    while index != data
        .len() {
            let current_indices = unravel_index(index, *self.shape);
            let axis_value = *current_indices[axis];

            if axis_value == 0 {
                if exclusive {
                    output_data.append(zero);
                } else {
                    output_data.append(*(data)[index]);
                }
            } else {
                let previous_axis_element_indices = replace_index(
                    current_indices, axis, axis_value - 1
                );
                let previous_axis_element_index = ravel_index(
                    *self.shape, previous_axis_element_indices
                );

                if exclusive {
                    output_data
                        .append(
                            *output_data[previous_axis_element_index]
                                + *(data)[previous_axis_element_index]
                        );
                } else {
                    output_data.append(*output_data[previous_axis_element_index] + *(data)[index]);
                };
            }

            index += 1;
        };

    TensorTrait::<T>::new(*self.shape, output_data.span())
}

/// Cf: TensorTrait::cumsum docstring
fn cumsum_reverse<
    T,
    impl TTensorTrait: TensorTrait<T>,
    impl TAddEq: AddEq<T>,
    impl TSub: Sub<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    self: @Tensor<T>, axis: usize, exclusive: Option<bool>, zero: T,
) -> Tensor<T> {
    let exclusive = match exclusive {
        Option::Some(val) => val,
        Option::None => false,
    };

    assert(axis < (*self.shape).len(), 'axis out of dimensions');
    let data = *self.data;
    let mut output_data = array![];
    let mut index: usize = 0;
    while index != data
        .len() {
            let current_indices = unravel_index(index, *self.shape);
            let mut axis_value = *current_indices[axis];

            if axis_value == 0 {
                // If the axis value is 0, we need to sum all the elements
                // in the axis.
                let mut sum = *(data)[index];
                if exclusive {
                    sum = zero;
                }

                let end_index = *(*self.shape)[axis] - 1;

                loop {
                    axis_value += 1;
                    if axis_value > end_index {
                        break ();
                    }

                    let next_axis_element_indices = replace_index(
                        current_indices, axis, axis_value
                    );
                    let next_axis_element_index = ravel_index(
                        *self.shape, next_axis_element_indices
                    );
                    sum += *data[next_axis_element_index];
                };

                output_data.append(sum);
            } else {
                // If the axis value is not 0, we only need to do a subtraction
                let previous_axis_element_indices = replace_index(
                    current_indices, axis, axis_value - 1
                );
                let previous_axis_element_index = ravel_index(
                    *self.shape, previous_axis_element_indices
                );

                if exclusive {
                    output_data.append(*output_data[previous_axis_element_index] - *(data)[index]);
                } else {
                    output_data
                        .append(
                            *output_data[previous_axis_element_index]
                                - *(data)[previous_axis_element_index]
                        );
                }
            }

            index += 1;
        };

    TensorTrait::<T>::new(*self.shape, output_data.span())
}
