use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;


impl FP16x16Sequence of SequenceTrait<FP16x16> {
    fn sequence_construct(tensors: Array<Tensor<FP16x16>>) -> Array<Tensor<FP16x16>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP16x16>> {
        functional::sequence_empty::sequence_empty::<FP16x16>()
    }
}
