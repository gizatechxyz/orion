pub mod span_fixed;

use orion_numbers::FixedTrait;

pub use span_fixed::FixedSpanMath;

pub trait SpanMathTrait<T> {
    fn arange(n: u32) -> Span<T>;
    fn dot(self: Span<T>, other: Span<T>) -> T;
    fn max(self: Span<T>) -> T;
    fn min(self: Span<T>) -> T;
    fn prod(self: Span<T>) -> T;
    fn sum(self: Span<T>) -> T;
}
