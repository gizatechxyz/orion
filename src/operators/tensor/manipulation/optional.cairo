use core::option::OptionTrait;
use orion::operators::tensor::{Tensor, TensorTrait};

/// Cf: TensorTrait::optional docstring
fn optional<
    T,
    +Copy<T>,
    +Drop<T>,
    impl TOption: OptionTrait<T>
>(
    self: @Tensor<T>
) -> Option<Tensor<T>> {
    Option::Some(*self)
}
