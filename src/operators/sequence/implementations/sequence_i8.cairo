use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::signed_integer::i8::i8;
use orion::operators::tensor::implementations::tensor_i8::I8Tensor;


impl I8Sequence of SequenceTrait<i8> {
    fn sequence_construct(tensors: Array<Tensor<i8>>) -> Array<Tensor<i8>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<i8>> {
        functional::sequence_empty::sequence_empty::<i8>()
    }
}
