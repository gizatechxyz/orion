// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod add {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::FixedTypeTensorAdd;
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = fp_tensor_1x3_helper();
            let tensor_2 = fp_tensor_1x3_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::FixedTypeTensorSub;
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = fp_tensor_1x3_helper();
            let tensor_2 = fp_tensor_1x3_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
            assert(*result.at(2) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::FixedTypeTensorMul;
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = fp_tensor_1x3_helper();
            let tensor_2 = fp_tensor_1x3_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorDiv, Tensor_fp
        };
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams, ravel_index, unravel_index};
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(100, false));
            data.append(FixedTrait::new_unscaled(200, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    #[cfg(test)]
    mod add {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorAdd, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = fp_tensor_2x2_helper();
            let tensor_2 = fp_tensor_2x2_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
            assert(*result.at(3) == FixedTrait::new_unscaled(6, false), 'result[3] = 6');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_add_broadcast() {
            let tensor_1 = fp_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(101, false), 'result[1] = 101');
            assert(*result.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12');
            assert(*result.at(3) == FixedTrait::new_unscaled(103, false), 'result[3] = 103');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(11, false), 'result[1] = 11');
            assert(*result.at(2) == FixedTrait::new_unscaled(102, false), 'result[2] = 102');
            assert(*result.at(3) == FixedTrait::new_unscaled(103, false), 'result[3] = 103');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorSub, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = fp_tensor_2x2_helper();
            let tensor_2 = fp_tensor_2x2_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
            assert(*result.at(2) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
            assert(*result.at(3) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub_broadcast() {
            let tensor_1 = fp_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(0, false));
            data.append(FixedTrait::new_unscaled(1, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
            assert(*result.at(2) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
            assert(*result.at(3) == FixedTrait::new_unscaled(2, false), 'result[3] = 2');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(0, false));
            data.append(FixedTrait::new_unscaled(1, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
            assert(*result.at(3) == FixedTrait::new_unscaled(2, false), 'result[3] = 2');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorMul, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = fp_tensor_2x2_helper();
            let tensor_2 = fp_tensor_2x2_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
            assert(*result.at(3) == FixedTrait::new_unscaled(9, false), 'result[3] = 9');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul_broadcast() {
            let tensor_1 = fp_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(100, false), 'result[1] = 100');
            assert(*result.at(2) == FixedTrait::new_unscaled(20, false), 'result[2] = 20');
            assert(*result.at(3) == FixedTrait::new_unscaled(300, false), 'result[3] = 300');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(10, false), 'result[1] = 10');
            assert(*result.at(2) == FixedTrait::new_unscaled(200, false), 'result[2] = 200');
            assert(*result.at(3) == FixedTrait::new_unscaled(300, false), 'result[3] = 300');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorDiv, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(100, false));
            data.append(FixedTrait::new_unscaled(200, false));
            data.append(FixedTrait::new_unscaled(300, false));
            data.append(FixedTrait::new_unscaled(400, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
            assert(*result.at(3) == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_div_broadcast() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(100, false));
            data.append(FixedTrait::new_unscaled(200, false));
            data.append(FixedTrait::new_unscaled(300, false));
            data.append(FixedTrait::new_unscaled(400, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
            assert(*result.at(2) == FixedTrait::new_unscaled(30, false), 'result[2] = 30');
            assert(*result.at(3) == FixedTrait::new_unscaled(4, false), 'result[3] = 4');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(20, false), 'result[1] = 20');
            assert(*result.at(2) == FixedTrait::new_unscaled(3, false), 'result[2] = 3');
            assert(*result.at(3) == FixedTrait::new_unscaled(4, false), 'result[3] = 4');
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    #[cfg(test)]
    mod add {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorAdd, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = fp_tensor_2x2x2_helper();
            let tensor_2 = fp_tensor_2x2x2_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(2, false), 'result[1] = 2');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
            assert(*result.at(3) == FixedTrait::new_unscaled(6, false), 'result[3] = 6');
            assert(*result.at(4) == FixedTrait::new_unscaled(8, false), 'result[4] = 8');
            assert(*result.at(5) == FixedTrait::new_unscaled(10, false), 'result[5] = 10');
            assert(*result.at(6) == FixedTrait::new_unscaled(12, false), 'result[6] = 12');
            assert(*result.at(7) == FixedTrait::new_unscaled(14, false), 'result[7] = 14');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_add_broadcast() {
            let tensor_1 = fp_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(11, false), 'result[1] = 11');
            assert(*result.at(2) == FixedTrait::new_unscaled(102, false), 'result[2] = 102');
            assert(*result.at(3) == FixedTrait::new_unscaled(103, false), 'result[3] = 103');
            assert(*result.at(4) == FixedTrait::new_unscaled(14, false), 'result[4] = 14');
            assert(*result.at(5) == FixedTrait::new_unscaled(15, false), 'result[5] = 15');
            assert(*result.at(6) == FixedTrait::new_unscaled(106, false), 'result[6] = 106');
            assert(*result.at(7) == FixedTrait::new_unscaled(107, false), 'result[7] = 107');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(11, false), 'result[1] = 11');
            assert(*result.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12');
            assert(*result.at(3) == FixedTrait::new_unscaled(13, false), 'result[3] = 13');
            assert(*result.at(4) == FixedTrait::new_unscaled(104, false), 'result[4] = 104');
            assert(*result.at(5) == FixedTrait::new_unscaled(105, false), 'result[5] = 105');
            assert(*result.at(6) == FixedTrait::new_unscaled(106, false), 'result[6] = 106');
            assert(*result.at(7) == FixedTrait::new_unscaled(107, false), 'result[7] = 107');

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(101, false), 'result[1] = 101');
            assert(*result.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12');
            assert(*result.at(3) == FixedTrait::new_unscaled(103, false), 'result[3] = 103');
            assert(*result.at(4) == FixedTrait::new_unscaled(14, false), 'result[4] = 14');
            assert(*result.at(5) == FixedTrait::new_unscaled(105, false), 'result[5] = 105');
            assert(*result.at(6) == FixedTrait::new_unscaled(16, false), 'result[6] = 16');
            assert(*result.at(7) == FixedTrait::new_unscaled(107, false), 'result[7] = 107');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorSub, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = fp_tensor_2x2x2_helper();
            let tensor_2 = fp_tensor_2x2x2_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(0, false), 'result[1] = 0');
            assert(*result.at(2) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
            assert(*result.at(3) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
            assert(*result.at(4) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
            assert(*result.at(5) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
            assert(*result.at(6) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
            assert(*result.at(7) == FixedTrait::new_unscaled(0, false), 'result[7] = 0');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub_broadcast() {
            let tensor_1 = fp_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(0, false));
            data.append(FixedTrait::new_unscaled(1, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
            assert(*result.at(3) == FixedTrait::new_unscaled(2, false), 'result[3] = 2');
            assert(*result.at(4) == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
            assert(*result.at(5) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
            assert(*result.at(6) == FixedTrait::new_unscaled(5, false), 'result[6] = 5');
            assert(*result.at(7) == FixedTrait::new_unscaled(6, false), 'result[7] = 6');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorMul, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = fp_tensor_2x2x2_helper();
            let tensor_2 = fp_tensor_2x2x2_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(4, false), 'result[2] = 4');
            assert(*result.at(3) == FixedTrait::new_unscaled(9, false), 'result[3] = 9');
            assert(*result.at(4) == FixedTrait::new_unscaled(16, false), 'result[4] = 16');
            assert(*result.at(5) == FixedTrait::new_unscaled(25, false), 'result[5] = 25');
            assert(*result.at(6) == FixedTrait::new_unscaled(36, false), 'result[6] = 36');
            assert(*result.at(7) == FixedTrait::new_unscaled(49, false), 'result[7] = 49');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul_broadcast() {
            let tensor_1 = fp_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
            assert(*result.at(1) == FixedTrait::new_unscaled(10, false), 'result[1] = 10');
            assert(*result.at(2) == FixedTrait::new_unscaled(200, false), 'result[2] = 200');
            assert(*result.at(3) == FixedTrait::new_unscaled(300, false), 'result[3] = 300');
            assert(*result.at(4) == FixedTrait::new_unscaled(40, false), 'result[4] = 40');
            assert(*result.at(5) == FixedTrait::new_unscaled(50, false), 'result[5] = 50');
            assert(*result.at(6) == FixedTrait::new_unscaled(600, false), 'result[6] = 600');
            assert(*result.at(7) == FixedTrait::new_unscaled(700, false), 'result[7] = 700');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_fp::{
            FixedTypeTensorDiv, Tensor_fp
        };
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(100, false));
            data.append(FixedTrait::new_unscaled(200, false));
            data.append(FixedTrait::new_unscaled(300, false));
            data.append(FixedTrait::new_unscaled(400, false));
            data.append(FixedTrait::new_unscaled(500, false));
            data.append(FixedTrait::new_unscaled(600, false));
            data.append(FixedTrait::new_unscaled(700, false));
            data.append(FixedTrait::new_unscaled(800, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(1, false), 'result[0] = 1');
            assert(*result.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
            assert(*result.at(2) == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
            assert(*result.at(3) == FixedTrait::new_unscaled(1, false), 'result[3] = 1');
            assert(*result.at(4) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
            assert(*result.at(5) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
            assert(*result.at(6) == FixedTrait::new_unscaled(1, false), 'result[6] = 1');
            assert(*result.at(7) == FixedTrait::new_unscaled(1, false), 'result[7] = 1');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_div_broadcast() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(100, false));
            data.append(FixedTrait::new_unscaled(200, false));
            data.append(FixedTrait::new_unscaled(300, false));
            data.append(FixedTrait::new_unscaled(400, false));
            data.append(FixedTrait::new_unscaled(500, false));
            data.append(FixedTrait::new_unscaled(600, false));
            data.append(FixedTrait::new_unscaled(700, false));
            data.append(FixedTrait::new_unscaled(800, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(FixedTrait::new_unscaled(10, false));
            data.append(FixedTrait::new_unscaled(100, false));
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == FixedTrait::new_unscaled(10, false), 'result[0] = 10');
            assert(*result.at(1) == FixedTrait::new_unscaled(20, false), 'result[1] = 20');
            assert(*result.at(2) == FixedTrait::new_unscaled(3, false), 'result[2] = 3');
            assert(*result.at(3) == FixedTrait::new_unscaled(4, false), 'result[3] = 4');
            assert(*result.at(4) == FixedTrait::new_unscaled(50, false), 'result[4] = 50');
            assert(*result.at(5) == FixedTrait::new_unscaled(60, false), 'result[5] = 60');
            assert(*result.at(6) == FixedTrait::new_unscaled(7, false), 'result[6] = 7');
            assert(*result.at(7) == FixedTrait::new_unscaled(8, false), 'result[7] = 8');
        }
    }
}
