// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 3, 'result[2] = 3');
    }

    #[test]
    #[available_gas(20000000)]
    fn changed_parameters() {
        // exclusive = false and reverse = false
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 3, 'result[2] = 3');

        // exclusive = false and reverse = true
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 3, 'result[0] = 3');
        assert((*result.data[1]).into() == 3, 'result[1] = 3');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');

        // exclusive = true and reverse = false
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 1, 'result[2] = 1');

        // exclusive = true and reverse = true
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 3, 'result[0] = 3');
        assert((*result.data[1]).into() == 2, 'result[1] = 2');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = u32_tensor_2x2_helper();
        // axis = 0
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 4, 'result[3] = 4');

        // axis = 1
        let result = tensor.cumsum(1, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 5, 'result[3] = 5');
    }


    #[test]
    #[available_gas(20000000)]
    fn changed_parameters() {
        //  axis = 0 exclusive = false and reverse = false
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 4, 'result[3] = 4');

        //  axis = 0 exclusive = false and reverse = true
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 4, 'result[1] = 4');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');

        //  axis = 0 exclusive = true and reverse = false
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 1, 'result[3] = 1');

        //  axis = 0 exclusive = true and reverse = true
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 3, 'result[1] = 3');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');

        //  axis = 1 exclusive = false and reverse = false
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 5, 'result[3] = 5');

        //  axis = 1 exclusive = false and reverse = true
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 5, 'result[2] = 5');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');

        //  axis = 1 exclusive = true and reverse = false
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 2, 'result[3] = 2');

        //  axis = 1 exclusive = true and reverse = true
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 3, 'result[2] = 3');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
    }
}


// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;


    #[test]
    #[available_gas(20000000)]
    fn default_parameters() {
        let tensor = u32_tensor_2x2x2_helper();

        // axis = 0
        let result = tensor.cumsum(0, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 6, 'result[5] = 6');
        assert((*result.data[6]).into() == 8, 'result[6] = 8');
        assert((*result.data[7]).into() == 10, 'result[7] = 10');

        // axis = 1
        let result = tensor.cumsum(1, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 4, 'result[3] = 4');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 10, 'result[6] = 10');
        assert((*result.data[7]).into() == 12, 'result[7] = 12');

        // axis = 2
        let result = tensor.cumsum(2, Option::None(()), Option::None(()));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 5, 'result[3] = 5');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 9, 'result[5] = 9');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 13, 'result[7] = 13');
    }


    #[test]
    #[available_gas(40000000)]
    fn changed_parameters() {
        //  axis = 0 exclusive = false and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 6, 'result[5] = 6');
        assert((*result.data[6]).into() == 8, 'result[6] = 8');
        assert((*result.data[7]).into() == 10, 'result[7] = 10');

        //  axis = 0 exclusive = false and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 4, 'result[0] = 4');
        assert((*result.data[1]).into() == 6, 'result[1] = 6');
        assert((*result.data[2]).into() == 8, 'result[2] = 8');
        assert((*result.data[3]).into() == 10, 'result[3] = 10');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');

        //  axis = 0 exclusive = true and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == 1, 'result[5] = 1');
        assert((*result.data[6]).into() == 2, 'result[6] = 2');
        assert((*result.data[7]).into() == 3, 'result[7] = 3');

        //  axis = 0 exclusive = true and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(0, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 4, 'result[0] = 4');
        assert((*result.data[1]).into() == 5, 'result[1] = 5');
        assert((*result.data[2]).into() == 6, 'result[2] = 6');
        assert((*result.data[3]).into() == 7, 'result[3] = 7');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == 0, 'result[5] = 0');
        assert((*result.data[6]).into() == 0, 'result[6] = 0');
        assert((*result.data[7]).into() == 0, 'result[7] = 0');

        //  axis = 1 exclusive = false and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 4, 'result[3] = 4');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 10, 'result[6] = 10');
        assert((*result.data[7]).into() == 12, 'result[7] = 12');

        //  axis = 1 exclusive = false and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 4, 'result[1] = 4');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 10, 'result[4] = 10');
        assert((*result.data[5]).into() == 12, 'result[5] = 12');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');

        //  axis = 1 exclusive = true and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 1, 'result[3] = 1');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == 0, 'result[5] = 0');
        assert((*result.data[6]).into() == 4, 'result[6] = 4');
        assert((*result.data[7]).into() == 5, 'result[7] = 5');

        //  axis = 1 exclusive = true and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(1, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 2, 'result[0] = 2');
        assert((*result.data[1]).into() == 3, 'result[1] = 3');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
        assert((*result.data[4]).into() == 6, 'result[4] = 6');
        assert((*result.data[5]).into() == 7, 'result[5] = 7');
        assert((*result.data[6]).into() == 0, 'result[6] = 0');
        assert((*result.data[7]).into() == 0, 'result[7] = 0');

        //  axis = 2 exclusive = false and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(false), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 5, 'result[3] = 5');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 9, 'result[5] = 9');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 13, 'result[7] = 13');

        //  axis = 2 exclusive = false and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(false), Option::Some(true));
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 5, 'result[2] = 5');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 9, 'result[4] = 9');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 13, 'result[6] = 13');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');

        //  axis = 2 exclusive = true and reverse = false
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(true), Option::Some(false));
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 0, 'result[2] = 0');
        assert((*result.data[3]).into() == 2, 'result[3] = 2');
        assert((*result.data[4]).into() == 0, 'result[4] = 0');
        assert((*result.data[5]).into() == 4, 'result[5] = 4');
        assert((*result.data[6]).into() == 0, 'result[6] = 0');
        assert((*result.data[7]).into() == 6, 'result[7] = 6');

        //  axis = 2 exclusive = true and reverse = true
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.cumsum(2, Option::Some(true), Option::Some(true));
        assert((*result.data[0]).into() == 1, 'result[0] = 1');
        assert((*result.data[1]).into() == 0, 'result[1] = 0');
        assert((*result.data[2]).into() == 3, 'result[2] = 3');
        assert((*result.data[3]).into() == 0, 'result[3] = 0');
        assert((*result.data[4]).into() == 5, 'result[4] = 5');
        assert((*result.data[5]).into() == 0, 'result[5] = 0');
        assert((*result.data[6]).into() == 7, 'result[6] = 7');
        assert((*result.data[7]).into() == 0, 'result[7] = 0');
    }
}
