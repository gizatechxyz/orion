use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;


impl I32Sequence of SequenceTrait<i32> {
    fn sequence_construct(tensors: Array<Tensor<i32>>) -> Array<Tensor<i32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<i32>> {
        functional::sequence_empty::sequence_empty::<i32>()
    }

    fn sequence_length(self: Array<Tensor<i32>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<i32>>, position: Tensor<i32>) -> Tensor<i32> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<i32>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<i32>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<i32>>, tensor: @Tensor<i32>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<i32>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<i32>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<i32> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
