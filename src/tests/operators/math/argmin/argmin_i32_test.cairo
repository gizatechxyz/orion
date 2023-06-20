
#[cfg(test)]
mod tensor1D_argmin_i32 {
    use array::{ArrayTrait,SpanTrait};
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {

        ////////////////////////////////////////////
        // case: default parameters 
        ////////////////////////////////////////////

        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(2_u32, false));
        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        
        let result = tensor.argmin(0,Option::None(()),Option::None(()));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(result.data.len() == 1, 'length == 1');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');


        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(2_u32, true));
        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        
        let result = tensor.argmin(0,Option::None(()),Option::None(()));
        assert(*result.data.at(0) == 2, 'result[0] = 2');
        assert(result.data.len() == 1, 'length == 1');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');

    }

    #[test]
    #[available_gas(20000000)]
    fn keepdims() {
        ////////////////////////////////////////////
        // case: keepdims == false
        ////////////////////////////////////////////

        // Shape should not change because keepdims
        // is forced to always true for 1d tensor 

        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(2_u32, true));
        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        
        let result = tensor.argmin(0,Option::Some(false),Option::None(()));
        assert(*result.data.at(0) == 2, 'result[0] = 2');
        assert(result.data.len() == 1, 'length == 1');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');
    }

    #[test]
    #[available_gas(20000000)]
    fn select_last_index() {

        ////////////////////////////////////////////
        // case: select_last_index == false 
        ////////////////////////////////////////////

        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1_u32, true));
        data.append(IntegerTrait::new(1_u32, true));
        data.append(IntegerTrait::new(1_u32, true));
        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        
        let result = tensor.argmin(0,Option::None(()),Option::Some(false));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(result.data.len() == 1, 'length == 1');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');

        ////////////////////////////////////////////
        // case: select_last_index == true 
        ////////////////////////////////////////////


        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1_u32, true));
        data.append(IntegerTrait::new(1_u32, true));
        data.append(IntegerTrait::new(1_u32, true));
        let extra = Option::<ExtraParams>::None(());

        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.argmin(0,Option::None(()),Option::Some(true));
        assert(*result.data.at(0) == 2, 'result[0] = 2');
        assert(result.data.len() == 1, 'length == 1');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');
    }
}



#[cfg(test)]
mod tensor2x2_argmin_i32 {
    use array::{ArrayTrait,SpanTrait};
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::operators::tensor::helpers::helpers_i32::{i32_tensor_1x3_helper,i32_tensor_2x2_helper, i32_tensor_2x2x2_helper};

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        ////////////////////////////////////////////
        // case: default parameters 
        ////////////////////////////////////////////

        let tensor = i32_tensor_2x2_helper();

        let result = tensor.argmin(0,Option::None(()),Option::None(()));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(result.data.len() == 2, 'length == 2');
        assert(result.shape.len() == 2, 'result.shape.len() == 2');


        let result = tensor.argmin(1,Option::None(()),Option::None(()));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(result.data.len() == 2, 'length == 2');
        assert(result.shape.len() == 2, 'result.shape.len() == 2');


    }

    #[test]
    #[available_gas(20000000)]
    fn keepdims() {
        
        ////////////////////////////////////////////
        // case: keepdims == false
        ////////////////////////////////////////////
        let tensor = i32_tensor_2x2_helper();

        let result = tensor.argmin(1,Option::Some(false),Option::None(()));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(result.data.len() == 2, 'length == 2');
        assert(result.shape.len() == 1, 'result.shape.len() == 1');

    }

    #[test]
    #[available_gas(20000000)]
    fn select_last_index() {

        ////////////////////////////////////////////
        // case: select_last_index == false
        ////////////////////////////////////////////
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.argmin(1,Option::None(()),Option::Some(false));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(result.data.len() == 2, 'length == 2');
        assert(result.shape.len() == 2, 'result.shape.len() == 2');

        ////////////////////////////////////////////
        // case: select_last_index == true
        ////////////////////////////////////////////
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.argmin(1,Option::None(()),Option::Some(true));
        assert(*result.data.at(0) == 1, 'result[0] = 1');
        assert(*result.data.at(1) == 1, 'result[1] = 1');
        assert(result.data.len() == 2, 'length == 2');
        assert(result.shape.len() == 2, 'result.shape.len() == 2');
    }
}


#[cfg(test)]
mod tensor2x2x2_argmin_i32 {
    use array::{ArrayTrait,SpanTrait};
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
    use orion::tests::operators::tensor::helpers::helpers_i32::{i32_tensor_1x3_helper,i32_tensor_2x2_helper, i32_tensor_2x2x2_helper};

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        ////////////////////////////////////////////
        // case: default parameters
        ////////////////////////////////////////////
        let tensor = i32_tensor_2x2x2_helper();

        let result = tensor.argmin(0,Option::None(()),Option::None(()));

        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(*result.data.at(2) == 0, 'result[2] = 0');
        assert(*result.data.at(3) == 0, 'result[3] = 0');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 3, 'result.shape.len() == 3');


        let result = tensor.argmin(1,Option::None(()),Option::None(()));

        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(*result.data.at(2) == 0, 'result[2] = 0');
        assert(*result.data.at(3) == 0, 'result[3] = 0');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 3, 'result.shape.len() == 3');


        let result = tensor.argmin(2,Option::None(()),Option::None(()));

        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(*result.data.at(2) == 0, 'result[2] = 0');
        assert(*result.data.at(3) == 0, 'result[3] = 0');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 3, 'result.shape.len() == 3');
    }


    #[test]
    #[available_gas(20000000)]
    fn keepdims() {
        
        ////////////////////////////////////////////
        // case: keepdims == false
        ////////////////////////////////////////////

        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.argmin(0,Option::Some(false),Option::None(()));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(*result.data.at(2) == 0, 'result[2] = 0');
        assert(*result.data.at(3) == 0, 'result[3] = 0');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 2, 'result.shape.len() == 2');



    }

    #[test]
    #[available_gas(20000000)]
    fn select_last_index() {

        ////////////////////////////////////////////
        // case: select_last_index == false
        ////////////////////////////////////////////
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));
        data.append(IntegerTrait::new(1_u32, false));

        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        let result = tensor.argmin(0,Option::None(()),Option::Some(false));
        assert(*result.data.at(0) == 0, 'result[0] = 0');
        assert(*result.data.at(1) == 0, 'result[1] = 0');
        assert(*result.data.at(2) == 0, 'result[2] = 0');
        assert(*result.data.at(3) == 0, 'result[3] = 0');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 3, 'result.shape.len() == 3');


        ////////////////////////////////////////////
        // case: select_last_index == true
        ////////////////////////////////////////////


        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
        let result = tensor.argmin(0,Option::None(()),Option::Some(true));
        assert(*result.data.at(0) == 1, 'result[0] = 1');
        assert(*result.data.at(1) == 1, 'result[1] = 1');
        assert(*result.data.at(2) == 1, 'result[2] = 1');
        assert(*result.data.at(3) == 1, 'result[3] = 1');
        assert(result.data.len() == 4, 'length == 4');
        assert(result.shape.len() == 3, 'result.shape.len() == 3');
    }
}
