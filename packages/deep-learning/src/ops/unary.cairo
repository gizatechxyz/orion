use orion_dl::Tensor;
use orion_numbers::FixedTrait;

pub(crate) fn tensor_log2<T, S, +FixedTrait<T, S>, +Copy<T>, +Drop<T>>(
    ref self: Tensor<T>
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(ele) => { result_data.append(FixedTrait::log2(*ele)); },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

pub(crate) fn tensor_exp2<T, S, +FixedTrait<T, S>, +Copy<T>, +Drop<T>>(
    ref self: Tensor<T>
) -> Tensor<T> {
    let mut result_data = ArrayTrait::new();

    loop {
        match self.data.pop_front() {
            Option::Some(ele) => { result_data.append(FixedTrait::exp2(*ele)); },
            Option::None(_) => { break; }
        };
    };

    Tensor { data: result_data.span() }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, tensor_log2, tensor_exp2};
    use orion_numbers::{F64, F64Impl, f64::helpers::assert_precise_span};

    #[test]
    fn test_tensor_log2() {
        let self_data: Array<F64> = array![
            F64Impl::new_unscaled(1),
            F64Impl::new_unscaled(2),
            F64Impl::new_unscaled(3),
            F64Impl::new_unscaled(4),
        ];
        let mut self = Tensor { data: self_data.span() };

        let result = tensor_log2(ref self);
        let expected = array![0, 4294967296, 6807362103, 8589934592].span();

        assert_precise_span(result.data, expected, 'Incorrect log2 result', Option::None);
    }

    #[test]
    fn test_tensor_exp2() {
        let self_data: Array<F64> = array![
            F64Impl::new_unscaled(1),
            F64Impl::new_unscaled(2),
            F64Impl::new_unscaled(3),
            F64Impl::new_unscaled(4),
        ];
        let mut self = Tensor { data: self_data.span() };

        let result = tensor_exp2(ref self);
        let expected = array![8589934592, 17179869184, 34359738368, 68719476736].span();

        assert_precise_span(result.data, expected, 'Incorrect exp2 result', Option::None);
    }
}
