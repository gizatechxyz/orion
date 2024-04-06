use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

/// Cf: NNTrait::hard_swish docstring
fn hard_swish<
    T,
    MAG,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Add<T>,
    +Div<T>,
    +Copy<T>,
    +Drop<T>,
    +Mul<T>,
    + Mul<Tensor<T>>,
    +Into<usize, MAG>,
>(
    mut x: Tensor<T>
) -> Tensor<T> {
    let x_cloned = x.clone();
    let mut data_result: Array<T> = array![];

    let a:usize = 6;
    let alpha = NumberTrait::<T, MAG>::one() / NumberTrait::<T, MAG>::new_unscaled(a.into(), false);
    let beta = NumberTrait::<T, MAG>::half();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let temp = (*item) * alpha + beta;
                let result = temp.min(NumberTrait::one()).max(NumberTrait::zero());
                data_result.append(result);
            },
            Option::None => { break; }
        };
    };

     return x_cloned * TensorTrait::new(x.shape, data_result.span());
}

