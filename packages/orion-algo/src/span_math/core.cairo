use core::array::ArrayTrait;
use core::option::OptionTrait;
use core::traits::TryInto;
use orion_numbers::f16x16::core::{f16x16, FixedTrait, ONE};
use orion_numbers::f16x16::core_trait::{I32Rem, I32Div};


use orion_algo::span_math::math;



#[generate_trait]
pub impl F16x16SpanMath of SpanMathTrait {
    fn arange(n: u32) -> Span<f16x16> {
        math::arange(n)
    }

    fn dot(self: Span<f16x16>, other: Span<f16x16>) -> f16x16 {
        math::dot(self, other)
    }

    fn max(self: Span<f16x16>) -> f16x16 {
        math::max(self)
    }

    fn min(self: Span<f16x16>) -> f16x16 {
        math::min(self)
    }

    fn prod(self: Span<f16x16>) -> f16x16 {
        math::prod(self)
    }

    fn sum(self: Span<f16x16>) -> f16x16 {
        math::sum(self)
    }

}