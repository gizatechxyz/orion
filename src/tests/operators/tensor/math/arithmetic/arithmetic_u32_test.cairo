// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod add {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::u32TensorAdd;
        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = u32_tensor_1x3_helper();
            let tensor_2 = u32_tensor_1x3_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 0_u32, 'result[0] = 0');
            assert(*result.at(1) == 2_u32, 'result[1] = 2');
            assert(*result.at(2) == 4_u32, 'result[2] = 4');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::u32TensorSub;
        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = u32_tensor_1x3_helper();
            let tensor_2 = u32_tensor_1x3_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0_u32, 'result[0] = 0');
            assert(*result.at(1) == 0_u32, 'result[1] = 0');
            assert(*result.at(2) == 0_u32, 'result[2] = 0');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::u32TensorMul;
        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = u32_tensor_1x3_helper();
            let tensor_2 = u32_tensor_1x3_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0_u32, 'result[0] = 0');
            assert(*result.at(1) == 1_u32, 'result[1] = 1');
            assert(*result.at(2) == 4_u32, 'result[2] = 4');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorDiv, Tensor_u32};
        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams, ravel_index, unravel_index};

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(100);
            data.append(200_u32);
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 1, 'result[0] = 1');
            assert(*result.at(1) == 1, 'result[1] = 1');
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

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorAdd, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = u32_tensor_2x2_helper();
            let tensor_2 = u32_tensor_2x2_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 2, 'result[1] = 2');
            assert(*result.at(2) == 4, 'result[2] = 4');
            assert(*result.at(3) == 6, 'result[3] = 6');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_add_broadcast() {
            let tensor_1 = u32_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 101, 'result[1] = 101');
            assert(*result.at(2) == 12, 'result[2] = 12');
            assert(*result.at(3) == 103, 'result[3] = 103');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 11, 'result[1] = 11');
            assert(*result.at(2) == 102, 'result[2] = 102');
            assert(*result.at(3) == 103, 'result[3] = 103');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorSub, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = u32_tensor_2x2_helper();
            let tensor_2 = u32_tensor_2x2_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 0, 'result[1] = 0');
            assert(*result.at(2) == 0, 'result[2] = 0');
            assert(*result.at(3) == 0, 'result[3] = 0');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub_broadcast() {
            let tensor_1 = u32_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(0);
            data.append(1);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 0, 'result[1] = 0');
            assert(*result.at(2) == 2, 'result[2] = 2');
            assert(*result.at(3) == 2, 'result[3] = 2');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(0);
            data.append(1);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 1, 'result[2] = 1');
            assert(*result.at(3) == 2, 'result[3] = 2');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorMul, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = u32_tensor_2x2_helper();
            let tensor_2 = u32_tensor_2x2_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 4, 'result[2] = 4');
            assert(*result.at(3) == 9, 'result[3] = 9');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul_broadcast() {
            let tensor_1 = u32_tensor_2x2_helper();
            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 100, 'result[1] = 100');
            assert(*result.at(2) == 20, 'result[2] = 20');
            assert(*result.at(3) == 300, 'result[3] = 300');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 10, 'result[1] = 10');
            assert(*result.at(2) == 200, 'result[2] = 200');
            assert(*result.at(3) == 300, 'result[3] = 300');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorDiv, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(100);
            data.append(200);
            data.append(300);
            data.append(400);
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 1, 'result[0] = 1');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 1, 'result[2] = 1');
            assert(*result.at(3) == 1, 'result[3] = 1');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_div_broadcast() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(100);
            data.append(200);
            data.append(300);
            data.append(400);
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 2, 'result[1] = 2');
            assert(*result.at(2) == 30, 'result[2] = 30');
            assert(*result.at(3) == 4, 'result[3] = 4');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10_u32);
            data.append(100_u32);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 20, 'result[1] = 20');
            assert(*result.at(2) == 3, 'result[2] = 3');
            assert(*result.at(3) == 4, 'result[3] = 4');
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

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorAdd, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_add() {
            let tensor_1 = u32_tensor_2x2x2_helper();
            let tensor_2 = u32_tensor_2x2x2_helper();

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 2, 'result[1] = 2');
            assert(*result.at(2) == 4, 'result[2] = 4');
            assert(*result.at(3) == 6, 'result[3] = 6');
            assert(*result.at(4) == 8, 'result[4] = 8');
            assert(*result.at(5) == 10, 'result[5] = 10');
            assert(*result.at(6) == 12, 'result[6] = 12');
            assert(*result.at(7) == 14, 'result[7] = 14');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_add_broadcast() {
            let tensor_1 = u32_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 11, 'result[1] = 11');
            assert(*result.at(2) == 102, 'result[2] = 102');
            assert(*result.at(3) == 103, 'result[3] = 103');
            assert(*result.at(4) == 14, 'result[4] = 14');
            assert(*result.at(5) == 15, 'result[5] = 15');
            assert(*result.at(6) == 106, 'result[6] = 106');
            assert(*result.at(7) == 107, 'result[7] = 107');

            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(1);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 11, 'result[1] = 11');
            assert(*result.at(2) == 12, 'result[2] = 12');
            assert(*result.at(3) == 13, 'result[3] = 13');
            assert(*result.at(4) == 104, 'result[4] = 104');
            assert(*result.at(5) == 105, 'result[5] = 105');
            assert(*result.at(6) == 106, 'result[6] = 106');
            assert(*result.at(7) == 107, 'result[7] = 107');

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(1);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 + tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 101, 'result[1] = 101');
            assert(*result.at(2) == 12, 'result[2] = 12');
            assert(*result.at(3) == 103, 'result[3] = 103');
            assert(*result.at(4) == 14, 'result[4] = 14');
            assert(*result.at(5) == 105, 'result[5] = 105');
            assert(*result.at(6) == 16, 'result[6] = 16');
            assert(*result.at(7) == 107, 'result[7] = 107');
        }
    }

    #[cfg(test)]
    mod sub {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorSub, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub() {
            let tensor_1 = u32_tensor_2x2x2_helper();
            let tensor_2 = u32_tensor_2x2x2_helper();

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 0, 'result[1] = 0');
            assert(*result.at(2) == 0, 'result[2] = 0');
            assert(*result.at(3) == 0, 'result[3] = 0');
            assert(*result.at(4) == 0, 'result[4] = 0');
            assert(*result.at(5) == 0, 'result[5] = 0');
            assert(*result.at(6) == 0, 'result[6] = 0');
            assert(*result.at(7) == 0, 'result[7] = 0');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_sub_broadcast() {
            let tensor_1 = u32_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(0);
            data.append(1);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 - tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 1, 'result[2] = 1');
            assert(*result.at(3) == 2, 'result[3] = 2');
            assert(*result.at(4) == 4, 'result[4] = 4');
            assert(*result.at(5) == 5, 'result[5] = 5');
            assert(*result.at(6) == 5, 'result[6] = 5');
            assert(*result.at(7) == 6, 'result[7] = 6');
        }
    }

    #[cfg(test)]
    mod mul {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorMul, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul() {
            let tensor_1 = u32_tensor_2x2x2_helper();
            let tensor_2 = u32_tensor_2x2x2_helper();

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 4, 'result[2] = 4');
            assert(*result.at(3) == 9, 'result[3] = 9');
            assert(*result.at(4) == 16, 'result[4] = 16');
            assert(*result.at(5) == 25, 'result[5] = 25');
            assert(*result.at(6) == 36, 'result[6] = 36');
            assert(*result.at(7) == 49, 'result[7] = 49');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_mul_broadcast() {
            let tensor_1 = u32_tensor_2x2x2_helper();

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 * tensor_2).data;

            assert(*result.at(0) == 0, 'result[0] = 0');
            assert(*result.at(1) == 10, 'result[1] = 10');
            assert(*result.at(2) == 200, 'result[2] = 200');
            assert(*result.at(3) == 300, 'result[3] = 300');
            assert(*result.at(4) == 40, 'result[4] = 40');
            assert(*result.at(5) == 50, 'result[5] = 50');
            assert(*result.at(6) == 600, 'result[6] = 600');
            assert(*result.at(7) == 700, 'result[7] = 700');
        }
    }

    #[cfg(test)]
    mod div {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::implementations::impl_tensor_u32::{u32TensorDiv, Tensor_u32};
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};

        #[test]
        #[available_gas(20000000)]
        fn tensor_div() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(100);
            data.append(200);
            data.append(300);
            data.append(400);
            data.append(500);
            data.append(600);
            data.append(700);
            data.append(800);
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 1, 'result[0] = 1');
            assert(*result.at(1) == 1, 'result[1] = 1');
            assert(*result.at(2) == 1, 'result[2] = 1');
            assert(*result.at(3) == 1, 'result[3] = 1');
            assert(*result.at(4) == 1, 'result[4] = 1');
            assert(*result.at(5) == 1, 'result[5] = 1');
            assert(*result.at(6) == 1, 'result[6] = 1');
            assert(*result.at(7) == 1, 'result[7] = 1');
        }

        #[test]
        #[available_gas(20000000)]
        fn tensor_div_broadcast() {
            let mut sizes = ArrayTrait::new();
            sizes.append(2);
            sizes.append(2);
            sizes.append(2);
            let mut data = ArrayTrait::new();
            data.append(100);
            data.append(200);
            data.append(300);
            data.append(400);
            data.append(500);
            data.append(600);
            data.append(700);
            data.append(800);
            let extra = Option::<ExtraParams>::None(());
            let tensor_1 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let mut sizes = ArrayTrait::new();
            sizes.append(1);
            sizes.append(2);
            sizes.append(1);
            let mut data = ArrayTrait::new();
            data.append(10);
            data.append(100);
            let extra = Option::<ExtraParams>::None(());
            let tensor_2 = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

            let result = (tensor_1 / tensor_2).data;

            assert(*result.at(0) == 10, 'result[0] = 10');
            assert(*result.at(1) == 20, 'result[1] = 20');
            assert(*result.at(2) == 3, 'result[2] = 3');
            assert(*result.at(3) == 4, 'result[3] = 4');
            assert(*result.at(4) == 50, 'result[4] = 50');
            assert(*result.at(5) == 60, 'result[5] = 60');
            assert(*result.at(6) == 7, 'result[6] = 7');
            assert(*result.at(7) == 8, 'result[7] = 8');
        }
    }
}
