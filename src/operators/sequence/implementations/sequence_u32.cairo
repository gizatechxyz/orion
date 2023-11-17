use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::operators::tensor::implementations::tensor_u32::U32Tensor;


impl U32Sequence of SequenceTrait<u32> {
    fn sequence_construct(tensors: Array<Tensor<u32>>) -> Array<Tensor<u32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<u32>> {
        functional::sequence_empty::sequence_empty::<u32>()
    }
}
