use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;


impl I32Sequence of SequenceTrait<i32> {
    fn sequence_construct(tensors: Array<Tensor<i32>>) -> Array<Tensor<i32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<i32>> {
        functional::sequence_empty::sequence_empty::<i32>()
    }
}
