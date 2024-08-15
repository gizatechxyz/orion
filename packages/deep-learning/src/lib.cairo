pub(crate) mod ops;
pub(crate) mod utils;

pub use ops::binary::{BinaryOpMetadata, tensor_add, tensor_mul, tensor_rem, tensor_lt};
pub use ops::unary::{tensor_log2, tensor_exp2};

#[derive(Drop, Copy)]
pub struct Tensor<T> {
    pub data: Span<T>,
}
