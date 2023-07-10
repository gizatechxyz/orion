// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {

    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl,FP8x23Into,FP8x23PartialEq};
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_1x3_helper;
    use orion::operators::tensor::core::TensorTrait;



    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));

        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(3,false), 'result[2] = 3');
    }

    #[test]
    #[available_gas(20000000)]
    fn changed_parameters() {

        // exclusive = false and reverse = false
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(3,false), 'result[2] = 3');

        // exclusive = false and reverse = true
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(3,false), 'result[0] = 3');
        assert((*result.data[1]) == FixedTrait::new_unscaled(3,false), 'result[1] = 3');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');

        // exclusive = true and reverse = false
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(1,false), 'result[2] = 1');

        // exclusive = true and reverse = true
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(3,false), 'result[0] = 3');
        assert((*result.data[1]) == FixedTrait::new_unscaled(2,false), 'result[1] = 2');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');

    }

}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl,FP8x23Into,FP8x23PartialEq};
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2_helper;
    use orion::operators::tensor::core::TensorTrait;

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = fp_tensor_2x2_helper();
        // axis = 0
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(4,false), 'result[3] = 4');

        // axis = 1
        let result = tensor.cumsum(1, Option::None(()), Option::None(()));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(5,false), 'result[3] = 5');
    }

   
    #[test]
    #[available_gas(20000000)]
    fn changed_parameters() {

        //  axis = 0 exclusive = false and reverse = false
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(4,false), 'result[3] = 4');

        //  axis = 0 exclusive = false and reverse = true
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(2,false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4,false), 'result[1] = 4');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');

        //  axis = 0 exclusive = true and reverse = false
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1,false), 'result[3] = 1');

        //  axis = 0 exclusive = true and reverse = true
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(2,false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(3,false), 'result[1] = 3');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0,false), 'result[3] = 0');





        //  axis = 1 exclusive = false and reverse = false
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(5,false), 'result[3] = 5');

        //  axis = 1 exclusive = false and reverse = true
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(1,false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(5,false), 'result[2] = 5');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');

        //  axis = 1 exclusive = true and reverse = false
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(2,false), 'result[3] = 2');

        //  axis = 1 exclusive = true and reverse = true
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(1,false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(3,false), 'result[2] = 3');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0,false), 'result[3] = 0');

    }



}


// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
        
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
    use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl,FP8x23Into,FP8x23PartialEq};
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;
    use orion::operators::tensor::core::TensorTrait;


    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = fp_tensor_2x2x2_helper();

        // axis = 0
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(6,false), 'result[5] = 6');
        assert((*result.data[6]) == FixedTrait::new_unscaled(8,false), 'result[6] = 8');
        assert((*result.data[7]) == FixedTrait::new_unscaled(10,false),'result[7] = 10');


        // axis = 1
        let result = tensor.cumsum(1, Option::None(()), Option::None(()));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(4,false), 'result[3] = 4');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5,false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(10,false), 'result[6] = 10');
        assert((*result.data[7]) == FixedTrait::new_unscaled(12,false), 'result[7] = 12');


        // axis = 2
        let result = tensor.cumsum(2, Option::None(()), Option::None(()));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(5,false), 'result[3] = 5');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(9,false), 'result[5] = 9');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6,false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(13,false), 'result[7] = 13');
    }



    #[test]
    #[available_gas(40000000)]
    fn changed_parameters() {

        //  axis = 0 exclusive = false and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(6,false), 'result[5] = 6');
        assert((*result.data[6]) == FixedTrait::new_unscaled(8,false), 'result[6] = 8');
        assert((*result.data[7]) == FixedTrait::new_unscaled(10,false),'result[7] = 10');


        //  axis = 0 exclusive = false and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(4,false), 'result[0] = 4');
        assert((*result.data[1]) == FixedTrait::new_unscaled(6,false), 'result[1] = 6');
        assert((*result.data[2]) == FixedTrait::new_unscaled(8,false), 'result[2] = 8');
        assert((*result.data[3]) == FixedTrait::new_unscaled(10,false), 'result[3] = 10');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5,false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6,false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7,false), 'result[7] = 7');

        //  axis = 0 exclusive = true and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0,false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0,false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(1,false), 'result[5] = 1');
        assert((*result.data[6]) == FixedTrait::new_unscaled(2,false), 'result[6] = 2');
        assert((*result.data[7]) == FixedTrait::new_unscaled(3,false), 'result[7] = 3');

        //  axis = 0 exclusive = true and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(4,false), 'result[0] = 4');
        assert((*result.data[1]) == FixedTrait::new_unscaled(5,false), 'result[1] = 5');
        assert((*result.data[2]) == FixedTrait::new_unscaled(6,false), 'result[2] = 6');
        assert((*result.data[3]) == FixedTrait::new_unscaled(7,false), 'result[3] = 7');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0,false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0,false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0,false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0,false), 'result[7] = 0');






        //  axis = 1 exclusive = false and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(4,false), 'result[3] = 4');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5,false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(10,false), 'result[6] = 10');
        assert((*result.data[7]) == FixedTrait::new_unscaled(12,false), 'result[7] = 12');


        //  axis = 1 exclusive = false and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(2,false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(4,false), 'result[1] = 4');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(10,false), 'result[4] = 10');
        assert((*result.data[5]) == FixedTrait::new_unscaled(12,false), 'result[5] = 12');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6,false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7,false), 'result[7] = 7');


        //  axis = 1 exclusive = true and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(1,false), 'result[3] = 1');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0,false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0,false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(4,false), 'result[6] = 4');
        assert((*result.data[7]) == FixedTrait::new_unscaled(5,false), 'result[7] = 5');

        //  axis = 1 exclusive = true and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(2,false), 'result[0] = 2');
        assert((*result.data[1]) == FixedTrait::new_unscaled(3,false), 'result[1] = 3');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0,false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(6,false), 'result[4] = 6');
        assert((*result.data[5]) == FixedTrait::new_unscaled(7,false), 'result[5] = 7');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0,false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0,false), 'result[7] = 0');








        //  axis = 2 exclusive = false and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(false), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(2,false), 'result[2] = 2');
        assert((*result.data[3]) == FixedTrait::new_unscaled(5,false), 'result[3] = 5');
        assert((*result.data[4]) == FixedTrait::new_unscaled(4,false), 'result[4] = 4');
        assert((*result.data[5]) == FixedTrait::new_unscaled(9,false), 'result[5] = 9');
        assert((*result.data[6]) == FixedTrait::new_unscaled(6,false), 'result[6] = 6');
        assert((*result.data[7]) == FixedTrait::new_unscaled(13,false), 'result[7] = 13');

        //  axis = 2 exclusive = false and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(false), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(1,false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(1,false), 'result[1] = 1');
        assert((*result.data[2]) == FixedTrait::new_unscaled(5,false), 'result[2] = 5');
        assert((*result.data[3]) == FixedTrait::new_unscaled(3,false), 'result[3] = 3');
        assert((*result.data[4]) == FixedTrait::new_unscaled(9,false), 'result[4] = 9');
        assert((*result.data[5]) == FixedTrait::new_unscaled(5,false), 'result[5] = 5');
        assert((*result.data[6]) == FixedTrait::new_unscaled(13,false), 'result[6] = 13');
        assert((*result.data[7]) == FixedTrait::new_unscaled(7,false), 'result[7] = 7');

        //  axis = 2 exclusive = true and reverse = false
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(true), Option::Some(false));
        assert((*result.data[0]) == FixedTrait::new_unscaled(0,false), 'result[0] = 0');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(0,false), 'result[2] = 0');
        assert((*result.data[3]) == FixedTrait::new_unscaled(2,false), 'result[3] = 2');
        assert((*result.data[4]) == FixedTrait::new_unscaled(0,false), 'result[4] = 0');
        assert((*result.data[5]) == FixedTrait::new_unscaled(4,false), 'result[5] = 4');
        assert((*result.data[6]) == FixedTrait::new_unscaled(0,false), 'result[6] = 0');
        assert((*result.data[7]) == FixedTrait::new_unscaled(6,false), 'result[7] = 6');

        //  axis = 2 exclusive = true and reverse = true
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(true), Option::Some(true));
        assert((*result.data[0]) == FixedTrait::new_unscaled(1,false), 'result[0] = 1');
        assert((*result.data[1]) == FixedTrait::new_unscaled(0,false), 'result[1] = 0');
        assert((*result.data[2]) == FixedTrait::new_unscaled(3,false), 'result[2] = 3');
        assert((*result.data[3]) == FixedTrait::new_unscaled(0,false), 'result[3] = 0');
        assert((*result.data[4]) == FixedTrait::new_unscaled(5,false), 'result[4] = 5');
        assert((*result.data[5]) == FixedTrait::new_unscaled(0,false), 'result[5] = 0');
        assert((*result.data[6]) == FixedTrait::new_unscaled(7,false), 'result[6] = 7');
        assert((*result.data[7]) == FixedTrait::new_unscaled(0,false), 'result[7] = 0');

    }

}
