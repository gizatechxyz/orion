use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn sign<
    T,
    MAG,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl FTensor: TensorTrait<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    mut self: Tensor<T>
) -> Tensor<T> {
    let mut result: Array<T> = array![];

    loop {
        match self.data.pop_front() {
            Option::Some(item) => { result.append((*item).sign()); },
            Option::None => { break; }
        };
    };

    TensorTrait::new(self.shape, result.span())
}
