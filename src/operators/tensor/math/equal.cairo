use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::helpers::check_compatibility;

/// Cf: TensorTrait::equal docstring
fn equal<
    T,
    F,
    impl UsizeFTensor: TensorTrait<usize, F>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    y: @Tensor<T>, z: @Tensor<T>
) -> Tensor<usize> {
    check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<usize>::new();
    let (mut smaller, mut bigger) = if (*y.data).len() < (*z.data).len() {
        (y, z)
    } else {
        (z, y)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        if bigger_data.len() == 0 {
            break ();
        };

        let bigger_current_index = *bigger_data.pop_front().unwrap();
        let smaller_current_index = *smaller_data[smaller_index];

        if bigger_current_index == smaller_current_index {
            data_result.append(1);
        } else {
            data_result.append(0);
        };

        smaller_index = (1 + smaller_index) % smaller_data.len();
    };

    return TensorTrait::new(*bigger.shape, data_result.span());
}
