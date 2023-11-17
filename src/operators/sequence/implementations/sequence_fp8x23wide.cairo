use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp8x23wide::core::FP8x23W;
use orion::operators::tensor::implementations::tensor_fp8x23wide::FP8x23WTensor;


impl FP8x23WSequence of SequenceTrait<FP8x23W> {
    fn sequence_construct(tensors: Array<Tensor<FP8x23W>>) -> Array<Tensor<FP8x23W>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP8x23W>> {
        functional::sequence_empty::sequence_empty::<FP8x23W>()
    }
}
