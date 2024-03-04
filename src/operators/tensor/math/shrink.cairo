use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: TensorTrait::shrink docstring
fn shrink<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut self: Tensor<T>, bias: Option<T>, lambd: Option<T>
) -> Tensor<T> {
    let bias: T = if bias.is_some() {
        bias.unwrap()
    } else {
        NumberTrait::zero()
    };

    let lambd: T = if lambd.is_some() {
        lambd.unwrap()
    } else {
        NumberTrait::half()
    };

    let mut data_result: Array<T> = array![];

    loop {
        match self.data.pop_front() {
            Option::Some(item) => {
                if (*item) < lambd.neg() {
                    let mut y = NumberTrait::add(*item, bias);
                    data_result.append(y);
                } else if (*item) > lambd {
                    let mut y = NumberTrait::sub(*item, bias);
                    data_result.append(y);
                } else {
                    data_result.append(NumberTrait::zero());
                }
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(self.shape, data_result.span())
}
