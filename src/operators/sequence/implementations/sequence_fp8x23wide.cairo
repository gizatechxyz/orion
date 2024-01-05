use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp8x23wide::core::FP8x23W;
use orion::operators::tensor::implementations::tensor_fp8x23wide::FP8x23WTensor;
use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;


impl FP8x23WSequence of SequenceTrait<FP8x23W> {
    fn sequence_construct(tensors: Array<Tensor<FP8x23W>>) -> Array<Tensor<FP8x23W>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP8x23W>> {
        functional::sequence_empty::sequence_empty::<FP8x23W>()
    }

    fn sequence_length(self: Array<Tensor<FP8x23W>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<FP8x23W>>, position: Tensor<i32>) -> Tensor<FP8x23W> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<FP8x23W>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP8x23W>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<FP8x23W>>, tensor: @Tensor<FP8x23W>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP8x23W>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<FP8x23W>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<FP8x23W> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }

}
