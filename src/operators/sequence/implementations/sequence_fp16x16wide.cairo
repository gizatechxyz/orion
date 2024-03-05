use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp16x16wide::core::FP16x16W;
use orion::operators::tensor::implementations::tensor_fp16x16wide::FP16x16WTensor;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;

impl FP16x16WSequence of SequenceTrait<FP16x16W> {
    fn sequence_construct(tensors: Array<Tensor<FP16x16W>>) -> Array<Tensor<FP16x16W>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP16x16W>> {
        functional::sequence_empty::sequence_empty::<FP16x16W>()
    }

    fn sequence_length(self: Array<Tensor<FP16x16W>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<FP16x16W>>, position: Tensor<i32>) -> Tensor<FP16x16W> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<FP16x16W>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP16x16W>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<FP16x16W>>, tensor: @Tensor<FP16x16W>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP16x16W>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<FP16x16W>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<FP16x16W> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
