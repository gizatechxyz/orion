use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::operators::tensor::implementations::tensor_bool::BoolTensor;


impl BoolSequence of SequenceTrait<bool> {
    fn sequence_construct(tensors: Array<Tensor<bool>>) -> Array<Tensor<bool>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<bool>> {
        functional::sequence_empty::sequence_empty::<bool>()
    }
}
