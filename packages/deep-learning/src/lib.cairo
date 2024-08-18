pub(crate) mod ops;
pub(crate) mod utils;

pub use ops::binary::{BinaryOpMetadata, tensor_add, tensor_mul, tensor_rem, tensor_lt};
pub use ops::unary::{tensor_log2, tensor_exp2, tensor_sin, tensor_sqrt, tensor_recip};
pub use ops::reduce::{tensor_reduce_sum_1d, tensor_reduce_sum_nd, ReduceOpMetadata};


#[derive(Drop, Copy)]
pub struct Tensor<T> {
    pub data: Span<T>,
}

use orion_data_structures::vec::NullableVec;

pub struct MutTensor<T> {
    pub data: NullableVec<T>,
}

pub impl MutTensorDestruct<T, +Drop<T>> of Destruct<MutTensor<T>> {
    fn destruct(self: MutTensor<T>) nopanic {
        self.data.destruct()
    }
}
