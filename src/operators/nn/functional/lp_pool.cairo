use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::debug::PrintTrait;
use orion::operators::nn::functional::common_pool::{common_pool};
use orion::operators::nn::{AUTO_PAD, POOLING_TYPE};


/// Cf: NNTrait::lp_pool docstring
fn lp_pool<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
    +PrintTrait<T>,
>(
    X: @Tensor<T>,
    auto_pad: Option<AUTO_PAD>,
    ceil_mode: Option<usize>,
    dilations: Option<Span<usize>>,
    kernel_shape: Span<usize>,
    p: Option<usize>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
    count_include_pad: Option<usize>,
) -> Tensor<T> {
    let p = match p {
        Option::Some(p) => p,
        Option::None => 2,
    };
    let count_include_pad = match count_include_pad {
        Option::Some(count_include_pad) => count_include_pad,
        Option::None => 0,
    };

    let (power_average, _) = common_pool(
        POOLING_TYPE::LPPOOL,
        count_include_pad,
        X,
        auto_pad,
        ceil_mode,
        dilations,
        kernel_shape,
        pads,
        strides,
        p
    );

    return power_average;
}
