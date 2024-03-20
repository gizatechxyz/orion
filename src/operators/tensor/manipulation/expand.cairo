use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use orion::operators::tensor::helpers::check_compatibility;
//use orion::operators::nn::helpers::prod;

/// Cf: TensorTrait::expand docstring
fn expand<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +Mul<Tensor<T>>,>(
    X: @Tensor<T>, shape: Tensor<usize>,
) -> Tensor<T> {
    check_compatibility((*X).shape, shape.data);

    let mut ones = ArrayTrait::new();
    let dim = prod(shape.data);

    let mut i = 0;
    while i != dim {
        ones.append(NumberTrait::one());
        i += 1;
    };

    return *X * TensorTrait::new(shape.data, ones.span());
}

///from  orion::operators::nn::helpers::prod; -> delete when nn refactor merged
fn prod<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +MulEq<T>,>(
    mut a: Span<T>
) -> T {
    let mut prod = NumberTrait::one();
    loop {
        match a.pop_front() {
            Option::Some(v) => { prod *= *v; },
            Option::None => { break prod; }
        };
    }
}
