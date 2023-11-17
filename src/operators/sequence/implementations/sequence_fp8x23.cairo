use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;


impl FP8x23Sequence of SequenceTrait<FP8x23> {
    fn sequence_construct(tensors: Array<Tensor<FP8x23>>) -> Array<Tensor<FP8x23>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP8x23>> {
        functional::sequence_empty::sequence_empty::<FP8x23>()
    }
}
