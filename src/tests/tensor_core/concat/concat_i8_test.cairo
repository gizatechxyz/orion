// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
    use orion::tests::helpers::tensor::i8::{i8_tensor_1x3_helper,i8_tensor_1x3_neg_helper, i8_tensor_2x3_helper,
    i8_tensor_3x3_helper, i8_tensor_2x2_helper, i8_tensor_2x2_neg_helper, 
    i8_tensor_3x2_helper, i8_tensor_3x2x2_helper, i8_tensor_3x2x2_neg_helper,
    i8_tensor_2x2x2_helper, i8_tensor_2x2x2_neg_helper, i8_tensor_3x3x3_helper, i8_tensor_3x3x3_neg_helper};
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use debug::PrintTrait;

    fn i8_tensor_3x2x1_helper() -> Tensor<i8> {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);
        sizes.append(2);
        sizes.append(1);

        let mut data = ArrayTrait::new();
        
        data.append(i8  {mag: 0, sign: false});
        data.append(i8 {mag: 1, sign: false});
        data.append(i8 {mag: 2, sign: false});
        data.append(i8 {mag: 3, sign: false});
        data.append(i8 {mag: 4, sign: false});
        data.append(i8 {mag: 5, sign: false});


        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i8>::new(sizes.span(), data.span(), extra);

    return tensor;
}


    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_1x3() {
        let tensor1 = i8_tensor_1x3_helper();
        let tensor2 = i8_tensor_1x3_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
        assert((*result.data[4]).into() == -1, 'result[4] = -1');
        assert((*result.data[5]).into() == -2, 'result[5] = -2');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
    }

// ===== 2D ===== //

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_axis_0() {
        let tensor1 = i8_tensor_2x2_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == -1, 'result[5] = -1');
        assert((*result.data[6]).into() == -2, 'result[5] = -2');
        assert((*result.data[7]).into() == -3, 'result[5] = -3');


        assert((*result.shape.at(0)) == 4, 'shape[0] = 4');
        assert((*result.shape.at(1)) == 2, 'shape[0] = 2');
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2_axis_1() {
        let tensor1 = i8_tensor_2x2_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);

        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == -1,'result[3] = -1');
        assert((*result.data[4]).into() == 2, 'result[4] = 2');
        assert((*result.data[5]).into() == 3, 'result[5] = 3');
        assert((*result.data[6]).into() == -2,'result[6] = -2');
        assert((*result.data[7]).into() == -3,'result[7] = -3');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 4, 'shape[0] = 4');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x3_2x2__axis_1() {
        let tensor1 = i8_tensor_2x3_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
        assert((*result.data[4]).into() == -1, 'result[4] = -1');
        assert((*result.data[5]).into() == 3, 'result[5] = 3');
        assert((*result.data[6]).into() == 4, 'result[5] = 4');
        assert((*result.data[7]).into() == 5, 'result[5] = 5');
        assert((*result.data[8]).into() == -2, 'result[5] = -2');
        assert((*result.data[9]).into() == -3, 'result[5] = -3');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 5, 'shape[1] = 5');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x3_2x2__axis_0() {
         let tensor1 = i8_tensor_2x3_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2_2x2__axis_0() {
        let tensor1 = i8_tensor_3x2_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis:0);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 0, 'result[5] = 0');
        assert((*result.data[7]).into() == -1,'result[5] = -1');
        assert((*result.data[8]).into() == -2,'result[5] = -2');
        assert((*result.data[9]).into() == -3,'result[5] = -3');

        assert((*result.shape.at(0)) == 5, 'shape[0] = 5');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2_2x2__axis_1() {
         let tensor1 = i8_tensor_3x2_helper();
        let tensor2 = i8_tensor_2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

// ===== 3D ===== //

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_0() {
        let tensor1 = i8_tensor_2x2x2_helper();
        let tensor2 = i8_tensor_2x2x2_neg_helper();
        let tensor3 = i8_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');
        assert((*result.data[8]).into() == 0, 'result[8] = 8');
        assert((*result.data[9]).into() == -1,'result[9] = -1');
        assert((*result.data[10]).into() == -2, 'result[10] = -2');
        assert((*result.data[11]).into() == -3, 'result[11] = -3');
        assert((*result.data[12]).into() == -4, 'result[12] = -4');
        assert((*result.data[13]).into() == -5, 'result[13] = -5');
        assert((*result.data[14]).into() == -6, 'result[14] = -6');
        assert((*result.data[15]).into() == -7, 'result[15] = -7');
        assert((*result.data[16]).into() == 0, 'result[16] = 0');
        assert((*result.data[17]).into() == 1, 'result[17] = 1');
        assert((*result.data[18]).into() == 2, 'result[18] = 2');
        assert((*result.data[19]).into() == 3, 'result[19] = 3');
        assert((*result.data[20]).into() == 4, 'result[20] = 4');
        assert((*result.data[21]).into() == 5, 'result[21] = 5');
        assert((*result.data[22]).into() == 6, 'result[22] = 6');
        assert((*result.data[23]).into() == 7, 'result[23] = 7');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

         #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_1() {
        let tensor1 = i8_tensor_2x2x2_helper();
        let tensor2 = i8_tensor_2x2x2_neg_helper();
        let tensor3 = i8_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == -1,'result[5] = -1');
        assert((*result.data[6]).into() == -2,'result[6] = -2');
        assert((*result.data[7]).into() == -3,'result[7] = -3');
        assert((*result.data[8]).into() == 0, 'result[8] = 0');
        assert((*result.data[9]).into() == 1, 'result[9] = 1');
        assert((*result.data[10]).into() == 2,'result[10] = 2');
        assert((*result.data[11]).into() == 3,'result[11] = 3');
        assert((*result.data[12]).into() == 4,'result[12] = 4');
        assert((*result.data[13]).into() == 5,'result[13] = 5');
        assert((*result.data[14]).into() == 6,'result[14] = 6');
        assert((*result.data[15]).into() == 7,'result[15] = 7');
        assert((*result.data[16]).into() == -4,'result[16] = -4');
        assert((*result.data[17]).into() == -5,'result[17] = -5');
        assert((*result.data[18]).into() == -6,'result[18] = -6');
        assert((*result.data[19]).into() == -7,'result[19] = -7');
        assert((*result.data[20]).into() == 4,'result[20] = 4');
        assert((*result.data[21]).into() == 5,'result[21] = 5');
        assert((*result.data[22]).into() == 6,'result[22] = 6');
        assert((*result.data[23]).into() == 7,'result[23] = 7');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 6, 'shape[1] = 6');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_2x2x2__axis_2() {
        let tensor1 = i8_tensor_2x2x2_helper();
        let tensor2 = i8_tensor_2x2x2_neg_helper();
        let tensor3 = i8_tensor_2x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor3);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]).into() == 0,  'result[0] = 0');
        assert((*result.data[1]).into() == 1,  'result[1] = 1');
        assert((*result.data[2]).into() == 0,  'result[2] = 0');
        assert((*result.data[3]).into() == -1, 'result[3] = -1');
        assert((*result.data[4]).into() == 0,  'result[4] = 0');
        assert((*result.data[5]).into() == 1,  'result[5] = 1');
        assert((*result.data[6]).into() == 2,  'result[6] = 2');
        assert((*result.data[7]).into() == 3,  'result[7] = 3');
        assert((*result.data[8]).into() == -2, 'result[8] = -2');
        assert((*result.data[9]).into() == -3, 'result[9] = -3');
        assert((*result.data[10]).into() == 2,  'result[10] = 2');
        assert((*result.data[11]).into() == 3,  'result[11] = 3');
        assert((*result.data[12]).into() == 4,  'result[12] = 4');
        assert((*result.data[13]).into() == 5,  'result[13] = 5');
        assert((*result.data[14]).into() == -4, 'result[14] = -4');
        assert((*result.data[15]).into() == -5, 'result[15] = -5');
        assert((*result.data[16]).into() == 4,  'result[16] = 4');
        assert((*result.data[17]).into() == 5,  'result[17] = 5');
        assert((*result.data[18]).into() == 6,  'result[18] = 6');
        assert((*result.data[19]).into() == 7,  'result[19] = 7');
        assert((*result.data[20]).into() == -6, 'result[20] = -6');
        assert((*result.data[21]).into() == -7, 'result[21] = -7');
        assert((*result.data[22]).into() == 6,  'result[22] = 6');
        assert((*result.data[23]).into() == 7,  'result[23] = 7');

        assert((*result.shape.at(0)) == 2, 'shape[0] = 2');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 6, 'shape[1] = 6');
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2__axis_0() {
        let tensor1 = i8_tensor_3x2x2_helper();
        let tensor2 = i8_tensor_3x2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
        assert((*result.data[0]).into() == 0,  'result[0] = 0');
        assert((*result.data[1]).into() == 1,  'result[1] = 1');
        assert((*result.data[2]).into() == 2,  'result[2] = 2');
        assert((*result.data[3]).into() == 3,  'result[3] = 3');
        assert((*result.data[4]).into() == 4,  'result[4] = 4');
        assert((*result.data[5]).into() == 5,  'result[5] = 5');
        assert((*result.data[6]).into() == 6,  'result[6] = 6');
        assert((*result.data[7]).into() == 7,  'result[7] = 7');
        assert((*result.data[8]).into() == 8,  'result[8] = 8');
        assert((*result.data[9]).into() == 9,  'result[9] = 9');
        assert((*result.data[10]).into() == 10, 'result[10] = 10');
        assert((*result.data[11]).into() == 11, 'result[11] = 11');
        assert((*result.data[12]).into() == 0,  'result[12] = 0');
        assert((*result.data[13]).into() == -1, 'result[13] = -1');
        assert((*result.data[14]).into() == -2, 'result[14] = -2');
        assert((*result.data[15]).into() == -3, 'result[15] = -3');
        assert((*result.data[16]).into() == -4, 'result[16] = -4');
        assert((*result.data[17]).into() == -5, 'result[17] = -5');
        assert((*result.data[18]).into() == -6, 'result[18] = -6');
        assert((*result.data[19]).into() == -7, 'result[19] = -7');
        assert((*result.data[20]).into() == -8, 'result[20] = -8');
        assert((*result.data[21]).into() == -9, 'result[21] = -9');
        assert((*result.data[22]).into() == -10, 'result[22] = -10');
        assert((*result.data[23]).into() == -11, 'result[23] = -11');

        assert((*result.shape.at(0)) == 6, 'shape[0] = 6');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2__axis_1() {
        let tensor1 = i8_tensor_3x2x2_helper();
        let tensor2 = i8_tensor_3x2x2_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
        assert((*result.data[0]).into() == 0,  'result[0] = 0');
        assert((*result.data[1]).into() == 1,  'result[1] = 1');
        assert((*result.data[2]).into() == 2,  'result[2] = 2');
        assert((*result.data[3]).into() == 3,  'result[3] = 3');
        assert((*result.data[4]).into() == 0,  'result[4] = 0');
        assert((*result.data[5]).into() == -1, 'result[5] = -1');
        assert((*result.data[6]).into() == -2, 'result[6] = -2');
        assert((*result.data[7]).into() == -3, 'result[7] = -3');
        assert((*result.data[8]).into() == 4,  'result[8] = 4');
        assert((*result.data[9]).into() == 5,  'result[9] = 5');
        assert((*result.data[10]).into() == 6,  'result[10] = 6');
        assert((*result.data[11]).into() == 7,  'result[11] = 7');
        assert((*result.data[12]).into() == -4, 'result[12] = -4');
        assert((*result.data[13]).into() == -5, 'result[13] = -5');
        assert((*result.data[14]).into() == -6, 'result[14] = -6');
        assert((*result.data[15]).into() == -7, 'result[15] = -7');
        assert((*result.data[16]).into() == 8,  'result[16] = 8');
        assert((*result.data[17]).into() == 9,  'result[17] = 9');
        assert((*result.data[18]).into() == 10, 'result[18] = 10');
        assert((*result.data[19]).into() == 11, 'result[19] = 11');
        assert((*result.data[20]).into() == -8, 'result[20] = -8');
        assert((*result.data[21]).into() == -9, 'result[21] = -9');
        assert((*result.data[22]).into() == -10, 'result[22] = -10');
        assert((*result.data[23]).into() == -11, 'result[23] = -11');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 4, 'shape[1] = 4');
        assert((*result.shape.at(2)) == 2, 'shape[1] = 2');

    }

    #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_2() {
        let tensor1 = i8_tensor_3x2x2_helper();
        let tensor2 = i8_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 2, 'result[3] = 2');
        assert((*result.data[4]).into() == 3, 'result[4] = 3');
        assert((*result.data[5]).into() == 1, 'result[5] = 1');
        assert((*result.data[6]).into() == 4, 'result[6] = 4');
        assert((*result.data[7]).into() == 5, 'result[7] = 5');
        assert((*result.data[8]).into() == 2, 'result[8] = 2');
        assert((*result.data[9]).into() == 6, 'result[9] = 6');
        assert((*result.data[10]).into() == 7,  'result[10] = 7');
        assert((*result.data[11]).into() == 3,  'result[11] = 3');
        assert((*result.data[12]).into() == 8,  'result[12] = 8');
        assert((*result.data[13]).into() == 9,  'result[13] = 9');
        assert((*result.data[14]).into() == 4,  'result[14] = 4');
        assert((*result.data[15]).into() == 10, 'result[15] = 10');
        assert((*result.data[16]).into() == 11, 'result[16] = 11');
        assert((*result.data[17]).into() == 5,  'result[17] = 5');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 2, 'shape[1] = 2');
        assert((*result.shape.at(2)) == 3, 'shape[1] = 4');

    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_0() {
        let tensor1 = i8_tensor_3x2x2_helper();
        let tensor2 = i8_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 0);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x2x2_3x2x1__axis_1() {
        let tensor1 = i8_tensor_3x2x2_helper();
        let tensor2 = i8_tensor_3x2x1_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

     #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn dimension_not_the_same() {
        let tensor1 = i8_tensor_1x3_helper();
        let tensor2 = i8_tensor_2x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn out_of_bound() {
        let tensor1 = i8_tensor_1x3_helper();
        let tensor2 = i8_tensor_1x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_not_same_dimension_1_fail() {
        let tensor1 = i8_tensor_2x2_helper();
        let tensor2 = i8_tensor_1x3_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

    #[test]
    #[should_panic]
    #[available_gas(20000000)]
    fn tensor_not_same_dimension_2_fail() {
        let tensor1 = i8_tensor_2x2_helper();
        let tensor2 = i8_tensor_3x2x2_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);

        let result = TensorTrait::concat(tensors: data.span(), axis: 1);
    }

     #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x3x3__axis_2() {
        let tensor1 = i8_tensor_3x3x3_helper();
        let tensor2 = i8_tensor_3x3x3_neg_helper();


        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor1);
    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]).into() == 0,  'result[0] = 0');
        assert((*result.data[1]).into() == 1,  'result[1] = 1');
        assert((*result.data[2]).into() == 2,  'result[2] = 2');
        assert((*result.data[3]).into() == 0,  'result[3] = 0');
        assert((*result.data[4]).into() == -1, 'result[4] = -1');
        assert((*result.data[5]).into() == -2, 'result[5] = -2');
        assert((*result.data[6]).into() == 0,  'result[6] = 0');
        assert((*result.data[7]).into() == 1,  'result[7] = 1');
        assert((*result.data[8]).into() == 2,  'result[8] = 2');
        assert((*result.data[9]).into() == 3,  'result[9] = 3');
        assert((*result.data[10]).into() == 4,  'result[10] = 4');
        assert((*result.data[11]).into() == 5,  'result[11] = 5');
        assert((*result.data[12]).into() == -3, 'result[12] = -3');
        assert((*result.data[13]).into() == -4, 'result[13] = -4');
        assert((*result.data[14]).into() == -5, 'result[14] = -5');
        assert((*result.data[15]).into() == 3,  'result[15] = 3');
        assert((*result.data[16]).into() == 4,  'result[16] = 4');
        assert((*result.data[17]).into() == 5,  'result[17] = 5');
        assert((*result.data[18]).into() == 6,  'result[18] = 6');
        assert((*result.data[19]).into() == 7,  'result[19] = 7');
        assert((*result.data[20]).into() == 8,  'result[20] = 8');
        assert((*result.data[21]).into() == -6, 'result[21] = -6');
        assert((*result.data[22]).into() == -7, 'result[22] = -7');
        assert((*result.data[23]).into() == -8, 'result[23] = -8');
        assert((*result.data[24]).into() == 6,  'result[23] = 6');
        assert((*result.data[25]).into() == 7,  'result[23] = 7');
        assert((*result.data[26]).into() == 8,  'result[23] = 8');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 9, 'shape[1] = 9');
    }

        #[test]
    #[available_gas(20000000)]
    fn tensor_onehot_tensor_3x3x3__axis_2_2() {
        let tensor1 = i8_tensor_3x3x3_helper();
        let tensor2 = i8_tensor_3x3x3_neg_helper();

        let mut data = ArrayTrait::new();
        data.append(tensor1);
        data.append(tensor2);
        data.append(tensor1);
        data.append(tensor2);

    
        let result = TensorTrait::concat(tensors: data.span(), axis: 2);
        assert((*result.data[0]).into() == 0,  'result[0] = 0');
        assert((*result.data[1]).into() == 1,  'result[1] = 1');
        assert((*result.data[2]).into() == 2,  'result[2] = 2');
        assert((*result.data[3]).into() == 0,  'result[3] = 0');
        assert((*result.data[4]).into() == -1, 'result[4] = -1');
        assert((*result.data[5]).into() == -2, 'result[5] = -2');
        assert((*result.data[6]).into() == 0,  'result[6] = 0');
        assert((*result.data[7]).into() == 1,  'result[7] = 1');
        assert((*result.data[8]).into() == 2,  'result[8] = 2');
        assert((*result.data[9]).into() == 0,  'result[9] = 0');
        assert((*result.data[10]).into() == -1, 'result[10] = -1');
        assert((*result.data[11]).into() == -2, 'result[11] = -2');
        assert((*result.data[12]).into() == 3,  'result[12] = 3');
        assert((*result.data[13]).into() == 4,  'result[13] = 4');
        assert((*result.data[14]).into() == 5,  'result[14] = 5');
        assert((*result.data[15]).into() == -3, 'result[15] = 3');
        assert((*result.data[16]).into() == -4, 'result[16] = 4');
        assert((*result.data[17]).into() == -5, 'result[17] = 5');
        assert((*result.data[18]).into() == 3,  'result[18] = -3');
        assert((*result.data[19]).into() == 4,  'result[19] = -4');
        assert((*result.data[20]).into() == 5,  'result[20] = -5');
        assert((*result.data[21]).into() == -3, 'result[21] = 3');
        assert((*result.data[22]).into() == -4, 'result[22] = 4');
        assert((*result.data[23]).into() == -5, 'result[23] = 5');

        assert((*result.shape.at(0)) == 3, 'shape[0] = 3');
        assert((*result.shape.at(1)) == 3, 'shape[1] = 3');
        assert((*result.shape.at(2)) == 12, 'shape[1] = 3');
    }

}