use core::array::ArrayTrait;
use core::array::SpanTrait;
use orion::numbers::NumberTrait;
use core::option::OptionTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::math::max_in_tensor::max_in_tensor;

/// Cf: NNTrait::global_maxpool docstring
fn global_maxpool<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    tensor: @Tensor<T>
) -> Tensor<T> {
    assert((*tensor).shape.len() == 4, 'Must be a 4D tensor');

    let mut data = (*tensor).data;
    let mut global_max_vals = ArrayTrait::new();

    let mut accum = 0;

    let N = (*tensor).shape.at(0);
    let C = (*tensor).shape.at(1);
    let H = (*tensor).shape.at(2);
    let W = (*tensor).shape.at(3);

    // height * width
    let mut area = *H * *W;

    loop {
        let mut sub_tensor = ArrayTrait::new();

        loop {
            match data.pop_front() {
                Option::Some(data) => {
                    if accum % area == 0 {
                        break ();
                    } else {
                        sub_tensor.append(*data);
                        accum += 1;
                    }
                },
                Option::None => { break; },
            };
        };

        let sub_tensor_max = max_in_tensor::<T>(sub_tensor.span());

        global_max_vals.append(sub_tensor_max);
    };

    let singleton_dim: usize = 1;

    let result = TensorTrait::new(
        array![*N, *C, singleton_dim, singleton_dim].span(), global_max_vals.span()
    );

    return result;
}
