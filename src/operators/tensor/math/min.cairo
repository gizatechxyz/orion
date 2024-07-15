use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape, check_compatibility
};

/// Cf: TensorTrait::min docstring
fn min<
    T,
    MAG,
    impl TTensorTrait: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    tensors: Span<Tensor<T>>
) -> Tensor<T> {
    assert(tensors.len() >= 1, 'Input tensors must be >= 1');

    let first_tensor = *tensors.at(0);

    if tensors.len() == 1 {
        return first_tensor;
    }

    let mut min_shape: Span<usize> = first_tensor.shape;
    let mut min_data: Span<T> = first_tensor.data;

    let mut tensor_counter: usize = 1;

    while tensor_counter != tensors
        .len() {
            let mut new_min_data: Array<T> = array![];

            let mut current_tensor = *tensors.at(tensor_counter);

            let mut broadcasted_shape = broadcast_shape(min_shape, current_tensor.shape);

            let num_elements = len_from_shape(broadcasted_shape);
            let mut n: usize = 0;
            while n != num_elements {
                let mut indices_broadcasted = unravel_index(n, broadcasted_shape);

                let mut indices_self = broadcast_index_mapping(min_shape, indices_broadcasted);
                let mut indices_other = broadcast_index_mapping(
                    current_tensor.shape, indices_broadcasted
                );

                let mut min_value = NumberTrait::min(
                    *(min_data)[indices_self], *(current_tensor.data)[indices_other]
                );
                new_min_data.append(min_value);

                n += 1;
            };

            min_shape = broadcasted_shape;
            min_data = new_min_data.span();
            tensor_counter += 1;
        };

    TensorTrait::<T>::new(min_shape, min_data)
}
