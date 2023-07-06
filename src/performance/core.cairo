use orion::operators::tensor::core::Tensor;
use orion::numbers::fixed_point::core::FixedType;

/// Trait
///
/// quantize_linear - Quantizes a Tensor using symmetric quantization.
/// quantize_linear_from_fp - Quantizes a FixedType Tensor using symmetric quantization.
trait PerfomanceTrait<T, O> {
    fn quantize_linear(
        self: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>
    ) -> Tensor::<O>;
}
