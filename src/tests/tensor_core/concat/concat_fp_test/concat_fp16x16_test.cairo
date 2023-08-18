// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_16x16::{
        FP16x16Impl, FP16x16Into, FP16x16PartialEq
    };
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::{fp_tensor_1x3_helper,fp_tensor_1x3_neg_helper, fp_tensor_2x3_helper, 
    fp_tensor_3x3_helper, fp_tensor_2x2_helper, fp_tensor_2x2_neg_helper, 
    fp_tensor_3x2_helper, fp_tensor_3x2x2_helper, fp_tensor_3x2x2_neg_helper,
    fp_tensor_2x2x2_helper, fp_tensor_2x2x2_neg_helper, fp_tensor_3x3x3_helper, fp_tensor_3x3x3_neg_helper};

    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use debug::PrintTrait;

    fn fp_tensor_3x2x1_helper() -> Tensor<FixedType> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(2);
        sizes.append(1);

        let mut data = ArrayTrait::new();
        data.append(FixedTrait::new_unscaled(0, false));
        data.append(FixedTrait::new_unscaled(1, false));
        data.append(FixedTrait::new_unscaled(2, false));
        data.append(FixedTrait::new_unscaled(3, false));
        data.append(FixedTrait::new_unscaled(4, false));
        data.append(FixedTrait::new_unscaled(5, false));


        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);

    return tensor;
    }


    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3() {
        let tensor1 = fp_tensor_1x3_helper();
        let tensor2 = fp_tensor_1x3_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, true), 'result[4] = -1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
    }

// ===== 2D ===== //

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_axis_0() {
        let tensor1 = fp_tensor_2x2_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, true), 'result[5] = -1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, true), 'result[5] = -3');


        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_axis_1() {
        let tensor1 = fp_tensor_2x2_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);

        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1, true), 'result[3] = -1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(2, false), 'result[4] = 2');
        assert((*result.data[5]) == FixedTrait::new_unscaled(3, false), 'result[5] = 2');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, true), 'result[5] = -3');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, true), 'result[5] = -4');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 4, 'shape[0] = 4');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x3_2x2__axis_1() {
        let tensor1 = fp_tensor_2x3_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, true), 'result[4] = -1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(3, false), 'result[5] = 3');
        assert((*result.data[6]) == FixedTrait::new_unscaled(4, false), 'result[5] = 4');
        assert((*result.data[7]) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');
        assert((*result.data[9]) == FixedTrait::new_unscaled(3, true), 'result[5] = -3');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 5, 'shape[1] = 5');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x3_2x2__axis_0() {
         let tensor1 = fp_tensor_2x3_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2_2x2__axis_0() {
        let tensor1 = fp_tensor_3x2_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis:0);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(1, true), 'result[5] = -1');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');
        assert((*result.data[9]) == FixedTrait::new_unscaled(3, true), 'result[5] = -3');

        assert((*result.shape.at(0)) == 5, 'shape[0] = 5');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2_2x2__axis_1() {
         let tensor1 = fp_tensor_3x2_helper();
        let tensor2 = fp_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

// ===== 3D ===== //

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_0() {
        let tensor1 = fp_tensor_2x2x2_helper();
        let tensor2 = fp_tensor_2x2x2_neg_helper();
        let tensor3 = fp_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6, false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7, false), 'result[7] = 7');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 8');
        assert((*result.data[9]) == FixedTrait::new_unscaled(1, true), 'result[9] = 9');
        assert((*result.data[10]) == FixedTrait::new_unscaled(2, true), 'result[10] = 10');
        assert((*result.data[11]) == FixedTrait::new_unscaled(3, true), 'result[11] = 11');
        assert((*result.data[12]) == FixedTrait::new_unscaled(4, true), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(5, true), 'result[13] = -1');
        assert((*result.data[14]) == FixedTrait::new_unscaled(6, true), 'result[14] = -2');
        assert((*result.data[15]) == FixedTrait::new_unscaled(7, true), 'result[15] = -3');
        assert((*result.data[16]) == FixedTrait::new_unscaled(0, false), 'result[16] = -4');
        assert((*result.data[17]) == FixedTrait::new_unscaled(1, false), 'result[17] = -5');
        assert((*result.data[18]) == FixedTrait::new_unscaled(2, false), 'result[18] = -6');
        assert((*result.data[19]) == FixedTrait::new_unscaled(3, false), 'result[19] = -7');
        assert((*result.data[20]) == FixedTrait::new_unscaled(4, false), 'result[20] = -8');
        assert((*result.data[21]) == FixedTrait::new_unscaled(5, false), 'result[21] = -9');
        assert((*result.data[22]) == FixedTrait::new_unscaled(6, false), 'result[22] = -10');
        assert((*result.data[23]) == FixedTrait::new_unscaled(7, false), 'result[23] = -11');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

         #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_1() {
        let tensor1 = fp_tensor_2x2x2_helper();
        let tensor2 = fp_tensor_2x2x2_neg_helper();
        let tensor3 = fp_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, true), 'result[5] = -1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, true), 'result[6] = -2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, true), 'result[7] = -3');
        assert((*result.data[8]) == FixedTrait::new_unscaled(0, false), 'result[8] = 0');
        assert((*result.data[9]) == FixedTrait::new_unscaled(1, false), 'result[9] = 1');
        assert((*result.data[10]) == FixedTrait::new_unscaled(2, false), 'result[10] = 2');
        assert((*result.data[11]) == FixedTrait::new_unscaled(3, false), 'result[11] = 3');
        assert((*result.data[12]) == FixedTrait::new_unscaled(4, false), 'result[12] = 4');
        assert((*result.data[13]) == FixedTrait::new_unscaled(5, false), 'result[13] = 5');
        assert((*result.data[14]) == FixedTrait::new_unscaled(6, false), 'result[14] = 6');
        assert((*result.data[15]) == FixedTrait::new_unscaled(7, false), 'result[15] = 7');
        assert((*result.data[16]) == FixedTrait::new_unscaled(4, true), 'result[16] = -4');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, true), 'result[17] = -5');
        assert((*result.data[18]) == FixedTrait::new_unscaled(6, true), 'result[18] = -6');
        assert((*result.data[19]) == FixedTrait::new_unscaled(7, true), 'result[19] = -7');
        assert((*result.data[20]) == FixedTrait::new_unscaled(4, false), 'result[20] = 4');
        assert((*result.data[21]) == FixedTrait::new_unscaled(5, false), 'result[21] = 5');
        assert((*result.data[22]) == FixedTrait::new_unscaled(6, false), 'result[22] = 6');
        assert((*result.data[23]) == FixedTrait::new_unscaled(7, false), 'result[23] = 7');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 6, 'shape[1] = 6');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_2() {
        let tensor1 = fp_tensor_2x2x2_helper();
        let tensor2 = fp_tensor_2x2x2_neg_helper();
        let tensor3 = fp_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1, true), 'result[3] = -1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, false), 'result[6] = 2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, false), 'result[7] = 3');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, true), 'result[8] = -2');
        assert((*result.data[9]) == FixedTrait::new_unscaled(3, true), 'result[9] = -3');
        assert((*result.data[10]) == FixedTrait::new_unscaled(2, false), 'result[10] = 2');
        assert((*result.data[11]) == FixedTrait::new_unscaled(3, false), 'result[11] = 3');
        assert((*result.data[12]) == FixedTrait::new_unscaled(4, false), 'result[12] = 4');
        assert((*result.data[13]) == FixedTrait::new_unscaled(5, false), 'result[13] = 5');
        assert((*result.data[14]) == FixedTrait::new_unscaled(4, true), 'result[14] = -4');
        assert((*result.data[15]) == FixedTrait::new_unscaled(5, true), 'result[15] = -5');
        assert((*result.data[16]) == FixedTrait::new_unscaled(4, false), 'result[16] = 4');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, false), 'result[17] = 5');
        assert((*result.data[18]) == FixedTrait::new_unscaled(6, false), 'result[18] = 6');
        assert((*result.data[19]) == FixedTrait::new_unscaled(7, false), 'result[19] = 7');
        assert((*result.data[20]) == FixedTrait::new_unscaled(6, true), 'result[20] = -6');
        assert((*result.data[21]) == FixedTrait::new_unscaled(7, true), 'result[21] = -7');
        assert((*result.data[22]) == FixedTrait::new_unscaled(6, false), 'result[22] = 6');
        assert((*result.data[23]) == FixedTrait::new_unscaled(7, false), 'result[23] = 7');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 6, 'shape[1] = 6');
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2__axis_0() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6, false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7, false), 'result[7] = 7');
        assert((*result.data[8]) == FixedTrait::new_unscaled(8, false), 'result[8] = 8');
        assert((*result.data[9]) == FixedTrait::new_unscaled(9, false), 'result[9] = 9');
        assert((*result.data[10]) == FixedTrait::new_unscaled(10, false), 'result[10] = 10');
        assert((*result.data[11]) == FixedTrait::new_unscaled(11, false), 'result[11] = 11');
        assert((*result.data[12]) == FixedTrait::new_unscaled(0, false), 'result[12] = 0');
        assert((*result.data[13]) == FixedTrait::new_unscaled(1, true), 'result[13] = -1');
        assert((*result.data[14]) == FixedTrait::new_unscaled(2, true), 'result[14] = -2');
        assert((*result.data[15]) == FixedTrait::new_unscaled(3, true), 'result[15] = -3');
        assert((*result.data[16]) == FixedTrait::new_unscaled(4, true), 'result[16] = -4');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, true), 'result[17] = -5');
        assert((*result.data[18]) == FixedTrait::new_unscaled(6, true), 'result[18] = -6');
        assert((*result.data[19]) == FixedTrait::new_unscaled(7, true), 'result[19] = -7');
        assert((*result.data[20]) == FixedTrait::new_unscaled(8, true), 'result[20] = -8');
        assert((*result.data[21]) == FixedTrait::new_unscaled(9, true), 'result[21] = -9');
        assert((*result.data[22]) == FixedTrait::new_unscaled(10, true), 'result[22] = -10');
        assert((*result.data[23]) == FixedTrait::new_unscaled(11, true), 'result[23] = -11');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2__axis_1() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, true), 'result[5] = -1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, true), 'result[6] = -2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, true), 'result[7] = -3');
        assert((*result.data[8]) == FixedTrait::new_unscaled(4, false), 'result[8] = 4');
        assert((*result.data[9]) == FixedTrait::new_unscaled(5, false), 'result[9] = 5');
        assert((*result.data[10]) == FixedTrait::new_unscaled(6, false), 'result[10] = 6');
        assert((*result.data[11]) == FixedTrait::new_unscaled(7, false), 'result[11] = 7');
        assert((*result.data[12]) == FixedTrait::new_unscaled(4, true), 'result[12] = -4');
        assert((*result.data[13]) == FixedTrait::new_unscaled(5, true), 'result[13] = -5');
        assert((*result.data[14]) == FixedTrait::new_unscaled(6, true), 'result[14] = -6');
        assert((*result.data[15]) == FixedTrait::new_unscaled(7, true), 'result[15] = -7');
        assert((*result.data[16]) == FixedTrait::new_unscaled(8, false), 'result[16] = 8');
        assert((*result.data[17]) == FixedTrait::new_unscaled(9, false), 'result[17] = 9');
        assert((*result.data[18]) == FixedTrait::new_unscaled(10, false), 'result[18] = 10');
        assert((*result.data[19]) == FixedTrait::new_unscaled(11, false), 'result[19] = 11');
        assert((*result.data[20]) == FixedTrait::new_unscaled(8, true), 'result[20] = -8');
        assert((*result.data[21]) == FixedTrait::new_unscaled(9, true), 'result[21] = -9');
        assert((*result.data[22]) == FixedTrait::new_unscaled(10, true), 'result[22] = -10');
        assert((*result.data[23]) == FixedTrait::new_unscaled(11, true), 'result[23] = -11');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 4, 'shape[1] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2__axis_2() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(2, false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(3, false), 'result[5] = -1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2, false), 'result[6] = -2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3, false), 'result[7] = -3');
        assert((*result.data[8]) == FixedTrait::new_unscaled(4, false), 'result[8] = 4');
        assert((*result.data[9]) == FixedTrait::new_unscaled(5, false), 'result[9] = 5');
        assert((*result.data[10]) == FixedTrait::new_unscaled(4, false), 'result[10] = 6');
        assert((*result.data[11]) == FixedTrait::new_unscaled(5, false), 'result[11] = 7');
        assert((*result.data[12]) == FixedTrait::new_unscaled(6, false), 'result[12] = -4');
        assert((*result.data[13]) == FixedTrait::new_unscaled(7, false), 'result[13] = -5');
        assert((*result.data[14]) == FixedTrait::new_unscaled(6, false), 'result[14] = -6');
        assert((*result.data[15]) == FixedTrait::new_unscaled(7, false), 'result[15] = -7');
        assert((*result.data[16]) == FixedTrait::new_unscaled(8, false), 'result[16] = 8');
        assert((*result.data[17]) == FixedTrait::new_unscaled(9, false), 'result[17] = 9');
        assert((*result.data[18]) == FixedTrait::new_unscaled(8, false), 'result[18] = 10');
        assert((*result.data[19]) == FixedTrait::new_unscaled(9, false), 'result[19] = 11');
        assert((*result.data[20]) == FixedTrait::new_unscaled(10, false), 'result[20] = -8');
        assert((*result.data[21]) == FixedTrait::new_unscaled(11, false), 'result[21] = -9');
        assert((*result.data[22]) == FixedTrait::new_unscaled(10, false), 'result[22] = -10');
        assert((*result.data[23]) == FixedTrait::new_unscaled(11, false), 'result[23] = -11');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 4, 'shape[1] = 4');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_2() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0, false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(2, false), 'result[3] = -1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(3, false), 'result[4] = 2');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1, false), 'result[5] = 3');
        assert((*result.data[6]) == FixedTrait::new_unscaled(4, false), 'result[6] = -2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(5, false), 'result[7] = -3');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, false), 'result[8] = 4');
        assert((*result.data[9]) == FixedTrait::new_unscaled(6, false), 'result[9] = 5');
        assert((*result.data[10]) == FixedTrait::new_unscaled(7, false), 'result[10] = -4');
        assert((*result.data[11]) == FixedTrait::new_unscaled(3, false), 'result[11] = -5');
        assert((*result.data[12]) == FixedTrait::new_unscaled(8, false), 'result[12] = 6');
        assert((*result.data[13]) == FixedTrait::new_unscaled(9, false), 'result[13] = 7');
        assert((*result.data[14]) == FixedTrait::new_unscaled(4, false), 'result[14] = -6');
        assert((*result.data[15]) == FixedTrait::new_unscaled(10, false), 'result[15] = -7');
        assert((*result.data[16]) == FixedTrait::new_unscaled(11, false), 'result[16] = 8');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, false), 'result[17] = 9');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 3, 'shape[1] = 4');

    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_0() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_1() {
        let tensor1 = fp_tensor_3x2x2_helper();
        let tensor2 = fp_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

     #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn dimension_not_the_same() {
        let tensor1 = fp_tensor_1x3_helper();
        let tensor2 = fp_tensor_2x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn out_of_bound() {
        let tensor1 = fp_tensor_1x3_helper();
        let tensor2 = fp_tensor_1x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_not_same_dimension_1_fail() {
        let tensor1 = fp_tensor_2x2_helper();
        let tensor2 = fp_tensor_1x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_not_same_dimension_2_fail() {
        let tensor1 = fp_tensor_3x3x3_helper();
        let tensor2 = fp_tensor_3x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x3x3__axis_1() {
        let tensor1 = fp_tensor_3x3x3_helper();
        let tensor2 = fp_tensor_3x3x3_neg_helper();
        let tensor3 = fp_tensor_3x3x3_helper();


        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4, false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5, false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6, false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7, false), 'result[7] = 7');
        assert((*result.data[8]) == FixedTrait::new_unscaled(8, false), 'result[8] = 8');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, true), 'result[10] = -1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(2, true), 'result[11] = -2');
        assert((*result.data[12]) == FixedTrait::new_unscaled(3, true), 'result[12] = -3');
        assert((*result.data[13]) == FixedTrait::new_unscaled(4, true), 'result[13] = -4');
        assert((*result.data[14]) == FixedTrait::new_unscaled(5, true), 'result[14] = -5');
        assert((*result.data[15]) == FixedTrait::new_unscaled(6, true), 'result[15] = -6');
        assert((*result.data[16]) == FixedTrait::new_unscaled(7, true), 'result[16] = -7');
        assert((*result.data[17]) == FixedTrait::new_unscaled(8, true), 'result[17] = -8');
        assert((*result.data[18]) == FixedTrait::new_unscaled(0, false), 'result[18] = 1');
        assert((*result.data[19]) == FixedTrait::new_unscaled(1, false), 'result[19] = 2');
        assert((*result.data[20]) == FixedTrait::new_unscaled(2, false), 'result[20] = 3');
        assert((*result.data[21]) == FixedTrait::new_unscaled(3, false), 'result[21] = 4');
        assert((*result.data[22]) == FixedTrait::new_unscaled(4, false), 'result[22] = 5');
        assert((*result.data[23]) == FixedTrait::new_unscaled(5, false), 'result[23] = 6');
        assert((*result.data[24]) == FixedTrait::new_unscaled(6, false), 'result[23] = 7');
        assert((*result.data[25]) == FixedTrait::new_unscaled(7, false), 'result[23] = 8');
        assert((*result.data[26]) == FixedTrait::new_unscaled(8, false), 'result[23] = 9');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 9, 'shape[1] = 9');
        assert((*result.shape.at(2)) == 3, 'shape[1] = 3');

    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x3x3__axis_2() {
        let tensor1 = fp_tensor_3x3x3_helper();
        let tensor2 = fp_tensor_3x3x3_neg_helper();
        let tensor3 = fp_tensor_3x3x3_helper();


        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, true), 'result[4] = -1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(1, false), 'result[7] = 7');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, false), 'result[8] = 8');
        assert((*result.data[9]) == FixedTrait::new_unscaled(3, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(4, false), 'result[10] = -1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(5, false), 'result[11] = -2');
        assert((*result.data[12]) == FixedTrait::new_unscaled(3, true), 'result[12] = -3');
        assert((*result.data[13]) == FixedTrait::new_unscaled(4, true), 'result[13] = -4');
        assert((*result.data[14]) == FixedTrait::new_unscaled(5, true), 'result[14] = -5');
        assert((*result.data[15]) == FixedTrait::new_unscaled(3, false), 'result[15] = -6');
        assert((*result.data[16]) == FixedTrait::new_unscaled(4, false), 'result[16] = -7');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, false), 'result[17] = -8');
        assert((*result.data[18]) == FixedTrait::new_unscaled(6, false), 'result[18] = 1');
        assert((*result.data[19]) == FixedTrait::new_unscaled(7, false), 'result[19] = 2');
        assert((*result.data[20]) == FixedTrait::new_unscaled(8, false), 'result[20] = 3');
        assert((*result.data[21]) == FixedTrait::new_unscaled(6, true), 'result[21] = 4');
        assert((*result.data[22]) == FixedTrait::new_unscaled(7, true), 'result[22] = 5');
        assert((*result.data[23]) == FixedTrait::new_unscaled(8, true), 'result[23] = 6');
        assert((*result.data[24]) == FixedTrait::new_unscaled(6, false), 'result[23] = 7');
        assert((*result.data[25]) == FixedTrait::new_unscaled(7, false), 'result[23] = 8');
        assert((*result.data[26]) == FixedTrait::new_unscaled(8, false), 'result[23] = 9');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 9, 'shape[1] = 9');
    }

        #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x3x3__axis_2_2() {
        let tensor1 = fp_tensor_3x3x3_helper();
        let tensor2 = fp_tensor_3x3x3_neg_helper();
        let tensor3 = fp_tensor_3x3x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
        data.append(tensor2);

    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2, false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0, false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(1, true), 'result[4] = -1');
        assert((*result.data[5]) == FixedTrait::new_unscaled(2, true), 'result[5] = -2');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0, false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(1, false), 'result[7] = 1');
        assert((*result.data[8]) == FixedTrait::new_unscaled(2, false), 'result[8] = 2');
        assert((*result.data[9]) == FixedTrait::new_unscaled(0, false), 'result[9] = 0');
        assert((*result.data[10]) == FixedTrait::new_unscaled(1, true), 'result[10] = -1');
        assert((*result.data[11]) == FixedTrait::new_unscaled(2, true), 'result[11] = -2');
        assert((*result.data[12]) == FixedTrait::new_unscaled(3, false), 'result[12] = 3');
        assert((*result.data[13]) == FixedTrait::new_unscaled(4, false), 'result[13] = 4');
        assert((*result.data[14]) == FixedTrait::new_unscaled(5, false), 'result[14] = 5');
        assert((*result.data[15]) == FixedTrait::new_unscaled(3, true), 'result[15] = 3');
        assert((*result.data[16]) == FixedTrait::new_unscaled(4, true), 'result[16] = 4');
        assert((*result.data[17]) == FixedTrait::new_unscaled(5, true), 'result[17] = 5');
        assert((*result.data[18]) == FixedTrait::new_unscaled(3, false), 'result[18] = -3');
        assert((*result.data[19]) == FixedTrait::new_unscaled(4, false), 'result[19] = -4');
        assert((*result.data[20]) == FixedTrait::new_unscaled(5, false), 'result[20] = -5');
        assert((*result.data[21]) == FixedTrait::new_unscaled(3, true), 'result[21] = 3');
        assert((*result.data[22]) == FixedTrait::new_unscaled(4, true), 'result[22] = 4');
        assert((*result.data[23]) == FixedTrait::new_unscaled(5, true), 'result[23] = 5');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 12, 'shape[1] = 12');
    }


}