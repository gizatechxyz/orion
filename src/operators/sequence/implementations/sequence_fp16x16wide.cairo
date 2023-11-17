use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp16x16wide::core::FP16x16W;
use orion::operators::tensor::implementations::tensor_fp16x16wide::FP16x16WTensor;


impl FP16x16WSequence of SequenceTrait<FP16x16W> {
    fn sequence_construct(tensors: Array<Tensor<FP16x16W>>) -> Array<Tensor<FP16x16W>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP16x16W>> {
        functional::sequence_empty::sequence_empty::<FP16x16W>()
    }
}
