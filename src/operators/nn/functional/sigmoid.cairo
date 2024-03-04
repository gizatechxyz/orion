use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::sigmoid docstring
fn sigmoid<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = NumberTrait::one()
                    / (NumberTrait::one() + (*item * NumberTrait::neg_one()).exp());
                data_result.append(result);
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(z.shape, data_result.span())
}

