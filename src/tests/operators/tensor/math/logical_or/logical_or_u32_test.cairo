// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use array::{ArrayTrait};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};

    #[test]
    #[available_gas(2000000)]
    fn tensor_logical_or() {
        let mut sizes = ArrayTrait::new();
        sizes.append(4);

        let mut arr_1 = ArrayTrait::<u32>::new();
        arr_1.append(0);
        arr_1.append(1);
        arr_1.append(0);
        arr_1.append(50);

        let mut arr_2 = ArrayTrait::<u32>::new();
        arr_2.append(0);
        arr_2.append(1);
        arr_2.append(1);
        arr_2.append(100);

        let extra = Option::<ExtraParams>::None(());

        let tensor_a = TensorTrait::<u32>::new(sizes.span(), arr_1.span(), extra);
        let tensor_b = TensorTrait::<u32>::new(sizes.span(), arr_2.span(), extra);

        let result_a = tensor_a.logical_or(@tensor_b);
        assert(*result_a.data[0] == 0, 'result[0] = 0'); // 0 || 0 = false
        assert(*result_a.data[1] == 1, 'result[1] = 1'); // 1 || 1 = true
        assert(*result_a.data[2] == 1, 'result[2] = 1'); // 0 || 1 = true
        assert(*result_a.data[3] == 1, 'result[3] = 1'); // [>1] || [>1] = true

        assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

        let result_b = tensor_b.logical_or(@tensor_a);
        assert(*result_b.data[0] == 0, 'result[0] = 0');
        assert(*result_b.data[1] == 1, 'result[1] = 1');
        assert(*result_b.data[2] == 1, 'result[2] = 1');
        assert(*result_b.data[3] == 1, 'result[3] = 1');

        assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use array::{ArrayTrait};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};

    #[test]
    #[available_gas(2000000)]
    fn tensor_logical_or() {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(3);

        let mut arr_1 = ArrayTrait::<u32>::new();
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(100);

        let mut arr_2 = ArrayTrait::<u32>::new();
        arr_2.append(0);
        arr_2.append(0);
        arr_2.append(0);
        arr_2.append(1);
        arr_2.append(1);
        arr_2.append(5);
        arr_2.append(0);
        arr_2.append(0);
        arr_2.append(0);

        let extra = Option::<ExtraParams>::None(());

        let tensor_a = TensorTrait::<u32>::new(sizes.span(), arr_1.span(), extra);
        let tensor_b = TensorTrait::<u32>::new(sizes.span(), arr_2.span(), extra);

        let result_a = tensor_a.logical_or(@tensor_b);
        assert(*result_a.data[0] == 0, 'result[0] = 0'); // 0 || 0 = false
        assert(*result_a.data[1] == 0, 'result[1] = 0'); // 0 || 0 = false
        assert(*result_a.data[2] == 0, 'result[2] = 0'); // 0 || 0 = false
        assert(*result_a.data[3] == 1, 'result[3] = 1'); // 1 || 1 = true
        assert(*result_a.data[4] == 1, 'result[4] = 1'); // 1 || 1 = true
        assert(*result_a.data[5] == 1, 'result[5] = 1'); // 1 || [>1] = true
        assert(*result_a.data[6] == 1, 'result[6] = 1'); // 1 || 0 = true
        assert(*result_a.data[7] == 1, 'result[7] = 1'); // 1 || 0 = true
        assert(*result_a.data[8] == 1, 'result[8] = 1'); // [>1] || 0 = true

        assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

        let result_b = tensor_b.logical_or(@tensor_a);
        assert(*result_b.data[0] == 0, 'result[0] = 0');
        assert(*result_b.data[1] == 0, 'result[1] = 0');
        assert(*result_b.data[2] == 0, 'result[2] = 0');
        assert(*result_b.data[3] == 1, 'result[3] = 1');
        assert(*result_b.data[4] == 1, 'result[4] = 1');
        assert(*result_b.data[5] == 1, 'result[5] = 1');
        assert(*result_b.data[6] == 1, 'result[6] = 1');
        assert(*result_b.data[7] == 1, 'result[7] = 1');
        assert(*result_b.data[8] == 1, 'result[8] = 1');

        assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
    }

    #[test]
    #[available_gas(2000000)]
    fn tensor_logical_or_broadcast() {
        let mut sizes_1 = ArrayTrait::new();
        sizes_1.append(4);
        sizes_1.append(3);

        let mut sizes_2 = ArrayTrait::new();
        sizes_2.append(1);
        sizes_2.append(3);

        let mut arr_1 = ArrayTrait::<u32>::new();
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);

        let mut arr_2 = ArrayTrait::<u32>::new();
        arr_2.append(0);
        arr_2.append(1);
        arr_2.append(1);

        let extra = Option::<ExtraParams>::None(());

        let tensor_a = TensorTrait::<u32>::new(sizes_1.span(), arr_1.span(), extra);
        let tensor_b = TensorTrait::<u32>::new(sizes_2.span(), arr_2.span(), extra);

        let result_a = tensor_b.logical_or(@tensor_a);
        assert(*result_a.data[0] == 0, 'result[0] = 0');
        assert(*result_a.data[1] == 1, 'result[1] = 1');
        assert(*result_a.data[2] == 1, 'result[2] = 1');
        assert(*result_a.data[3] == 0, 'result[3] = 0');
        assert(*result_a.data[4] == 1, 'result[4] = 1');
        assert(*result_a.data[5] == 1, 'result[5] = 1');
        assert(*result_a.data[6] == 1, 'result[6] = 1');
        assert(*result_a.data[7] == 1, 'result[7] = 1');
        assert(*result_a.data[8] == 1, 'result[8] = 1');
        assert(*result_a.data[9] == 1, 'result[9] = 1');
        assert(*result_a.data[10] == 1, 'result[10] = 1');
        assert(*result_a.data[11] == 1, 'result[11] = 1');

        assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

        let result_b = tensor_a.logical_or(@tensor_b);
        assert(*result_b.data[0] == 0, 'result[0] = 0');
        assert(*result_b.data[1] == 1, 'result[1] = 1');
        assert(*result_b.data[2] == 1, 'result[2] = 1');
        assert(*result_b.data[3] == 0, 'result[3] = 0');
        assert(*result_b.data[4] == 1, 'result[4] = 1');
        assert(*result_b.data[5] == 1, 'result[5] = 1');
        assert(*result_b.data[6] == 1, 'result[6] = 1');
        assert(*result_b.data[7] == 1, 'result[7] = 1');
        assert(*result_b.data[8] == 1, 'result[8] = 1');
        assert(*result_b.data[9] == 1, 'result[9] = 1');
        assert(*result_b.data[10] == 1, 'result[10] = 1');
        assert(*result_b.data[11] == 1, 'result[11] = 1');

        assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use array::{ArrayTrait};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};

    #[test]
    #[available_gas(2000000)]
    fn tensor_logical_or() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut arr_1 = ArrayTrait::<u32>::new();
        arr_1.append(0_u32);
        arr_1.append(0_u32);
        arr_1.append(0_u32);
        arr_1.append(0_u32);
        arr_1.append(1_u32);
        arr_1.append(1_u32);
        arr_1.append(1_u32);
        arr_1.append(1_u32);

        let mut arr_2 = ArrayTrait::<u32>::new();
        arr_2.append(0_u32);
        arr_2.append(1_u32);
        arr_2.append(0_u32);
        arr_2.append(1_u32);
        arr_2.append(0_u32);
        arr_2.append(1_u32);
        arr_2.append(0_u32);
        arr_2.append(1_u32);

        let extra = Option::<ExtraParams>::None(());

        let tensor_a = TensorTrait::<u32>::new(sizes.span(), arr_1.span(), extra);
        let tensor_b = TensorTrait::<u32>::new(sizes.span(), arr_2.span(), extra);

        let result_a = tensor_a.logical_or(@tensor_b);
        assert(*result_a.data[0] == 0, 'result[0] = 0');
        assert(*result_a.data[1] == 1, 'result[1] = 1');
        assert(*result_a.data[2] == 0, 'result[2] = 0');
        assert(*result_a.data[3] == 1, 'result[3] = 1');
        assert(*result_a.data[4] == 1, 'result[4] = 1');
        assert(*result_a.data[5] == 1, 'result[5] = 1');
        assert(*result_a.data[6] == 1, 'result[6] = 1');
        assert(*result_a.data[7] == 1, 'result[7] = 1');

        assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

        let result_b = tensor_b.logical_or(@tensor_a);
        assert(*result_b.data[0] == 0, 'result[0] = 0');
        assert(*result_b.data[1] == 1, 'result[1] = 1');
        assert(*result_b.data[2] == 0, 'result[2] = 0');
        assert(*result_b.data[3] == 1, 'result[3] = 1');
        assert(*result_b.data[4] == 1, 'result[4] = 1');
        assert(*result_b.data[5] == 1, 'result[5] = 1');
        assert(*result_b.data[6] == 1, 'result[6] = 1');
        assert(*result_b.data[7] == 1, 'result[7] = 1');

        assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
    }

    #[test]
    #[available_gas(2000000)]
    fn tensor_logical_or_broadcast() {
        let mut sizes_1 = ArrayTrait::new();
        sizes_1.append(2);
        sizes_1.append(2);
        sizes_1.append(2);

        let mut sizes_2 = ArrayTrait::new();
        sizes_2.append(1);
        sizes_2.append(2);
        sizes_2.append(1);

        let mut arr_1 = ArrayTrait::<u32>::new();
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(0);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);
        arr_1.append(1);

        let mut arr_2 = ArrayTrait::<u32>::new();
        arr_2.append(0);
        arr_2.append(1);

        let extra = Option::<ExtraParams>::None(());

        let tensor_a = TensorTrait::<u32>::new(sizes_1.span(), arr_1.span(), extra);
        let tensor_b = TensorTrait::<u32>::new(sizes_2.span(), arr_2.span(), extra);

        let result_a = tensor_b.logical_or(@tensor_a);
        assert(*result_a.data[1] == 1, 'result[1] = 1');
        assert(*result_a.data[2] == 0, 'result[2] = 0');
        assert(*result_a.data[3] == 1, 'result[3] = 1');
        assert(*result_a.data[4] == 1, 'result[4] = 1');
        assert(*result_a.data[5] == 1, 'result[5] = 1');
        assert(*result_a.data[6] == 1, 'result[6] = 1');
        assert(*result_a.data[7] == 1, 'result[7] = 1');

        assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

        let result_b = tensor_a.logical_or(@tensor_b);
        assert(*result_b.data[0] == 0, 'result[0] = 0');
        assert(*result_b.data[1] == 1, 'result[1] = 1');
        assert(*result_b.data[2] == 0, 'result[2] = 0');
        assert(*result_b.data[3] == 1, 'result[3] = 1');
        assert(*result_b.data[4] == 1, 'result[4] = 1');
        assert(*result_b.data[5] == 1, 'result[5] = 1');
        assert(*result_b.data[6] == 1, 'result[6] = 1');
        assert(*result_b.data[7] == 1, 'result[7] = 1');

        assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
    }
}

