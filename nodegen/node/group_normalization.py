import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def _group_normalization(x, num_groups, scale, bias, epsilon=1e-5):
    # Assume channel is first dim
    assert x.shape[1] % num_groups == 0
    group_size = x.shape[1] // num_groups
    # Reshape to [N, group_size, C/group_size, H, W, ...]
    new_shape = [x.shape[0], num_groups, group_size, *list(x.shape[2:])]
    x_reshaped = x.reshape(new_shape)
    axes = tuple(range(2, len(new_shape)))
    mean = np.mean(x_reshaped, axis=axes, keepdims=True)
    var = np.var(x_reshaped, axis=axes, keepdims=True)
    x_normalized = ((x_reshaped - mean) / np.sqrt(var + epsilon)).reshape(x.shape)
    dim_ones = (1,) * (len(x.shape) - 2)
    scale = scale.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return scale * x_normalized + bias

class Group_normalization(RunAll):
    @staticmethod
    def group_normalization_fp16x16():
        def group_normalization_fp16x16_4D():
            c = 4
            num_groups = 2
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_4D"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)

        def group_normalization_fp16x16_4D_epsilon(): 
            c = 4
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_4D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        def group_normalization_fp16x16_4D_groups_equal_to_channels():
            c = 2
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon ).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_4D_groups_equal_to_channels"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        def group_normalization_fp16x16_4D_single_group():
            c = 3
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_4D_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)

        
        def group_normalization_fp16x16_3D(): 
            c = 2
            num_groups = 1
            x = np.random.randn(3, c, 2,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_3D"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)


        def group_normalization_fp16x16_3D_epsilon(): 
            c = 2
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, 2,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_3D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        def group_normalization_fp16x16_3D_groups_equal_to_channels():
            c = 2
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn(3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias,  epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_3D_groups_equal_to_channels"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)



        def group_normalization_fp16x16_3D_single_group():
            c = 2
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_3D_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        def group_normalization_fp16x16_2D(): 
            c = 2
            num_groups = 1
            x = np.random.randn(3, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_2D"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)


        def group_normalization_fp16x16_2D_epsilon(): 
            c = 2
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)

        
        def group_normalization_fp16x16_2D_single_group(): 
            c = 3
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_2D_single_group"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)

        def group_normalization_fp16x16_2D_groups_equal_to_channels(): 
            c = 2
            num_groups = 2
            x = np.random.randn(3, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_2D_groups_equal_to_channels"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)

        def group_normalization_fp16x16_highdim(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 2
            num_groups = 1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "group_normalization_fp16x16_highdim"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)


        def group_normalization_fp16x16_highdim_epsilon(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 2
            num_groups = 1
            epsilon = 0.1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_highdim_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)
    
        def group_normalization_fp16x16_highdim_groups_equal_to_channels(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 3
            num_groups = 3
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "group_normalization_fp16x16_highdim_groups_equal_to_channels"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()))", name)
    
    
        def group_normalization_fp16x16_highdim_single_group(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 3
            num_groups = 1
            epsilon = 0.1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "group_normalization_fp16x16_highdim_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        group_normalization_fp16x16_4D()    
        group_normalization_fp16x16_4D_epsilon()    
        group_normalization_fp16x16_4D_groups_equal_to_channels() 
        group_normalization_fp16x16_4D_single_group()    

        group_normalization_fp16x16_3D() 
        group_normalization_fp16x16_3D_epsilon() 
        group_normalization_fp16x16_3D_groups_equal_to_channels()    
        group_normalization_fp16x16_3D_single_group() 

        group_normalization_fp16x16_2D() 
        group_normalization_fp16x16_2D_epsilon() 
        group_normalization_fp16x16_2D_groups_equal_to_channels()    
        group_normalization_fp16x16_2D_single_group() 

        group_normalization_fp16x16_highdim()
        group_normalization_fp16x16_highdim_epsilon()
        group_normalization_fp16x16_highdim_groups_equal_to_channels()
        group_normalization_fp16x16_highdim_single_group()



    @staticmethod
    def group_normalization_fp8x23():
        def group_normalization_fp8x23_4D():
            c = 4
            num_groups = 2
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_4D"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()))", name)
    
        def group_normalization_fp8x23_4D_epsilon(): 
            c = 4
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_4D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    
    
        def group_normalization_fp8x23_4D_groups_equal_to_channels():
            c = 2
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias , epsilon ).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_4D_groups_equal_to_channels"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

    
    
        def group_normalization_fp8x23_4D_single_group():
            c = 3
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn(3, c, 2, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_4D_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
   


        def group_normalization_fp8x23_3D():
            c = 4
            num_groups = 2
            x = np.random.randn( 3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias,).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_3D"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()) )", name)
    
        def group_normalization_fp8x23_3D_epsilon():
            c = 4
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn( 3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_3D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    
    
        def group_normalization_fp8x23_3D_groups_equal_to_channels():
            c = 2
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn( 3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_3D_groups_equal_to_channels" 
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 ,  Option::Some( FixedTrait::new(838860, false)) )", name)
    
    
        def group_normalization_fp8x23_3D_single_group():
            c = 3
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn( 3, c, 2).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_3D_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)



        def group_normalization_fp8x23_2D():
            c = 4
            num_groups = 2
            x = np.random.randn( 3, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_2D"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()))", name)
    
    
        def group_normalization_fp8x23_2D_epsilon():
            c = 4
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn( 3, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    

        def group_normalization_fp8x23_2D_groups_equal_to_channels():
            c = 2
            num_groups = 2
            epsilon =  1e-1
            x = np.random.randn( 3, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_2D_groups_equal_to_channels"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)


        def group_normalization_fp8x23_2D_single_group():
            c = 3
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn( 3, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_2D_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)


        def group_normalization_fp8x23_highdim(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 2
            num_groups = 1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_highdim"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()))", name)


        def group_normalization_fp8x23_highdim_epsilon(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 2
            num_groups = 1
            epsilon = 0.1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_highdim_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

    
        def group_normalization_fp8x23_highdim_groups_equal_to_channels(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 3
            num_groups = 3
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_highdim_groups_equal_to_channels"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::None(()))", name)
    
    
        def group_normalization_fp8x23_highdim_single_group(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,3)
            c = 2
            num_groups = 1
            epsilon =  1e-1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "group_normalization_fp8x23_highdim_single_group"
            make_test([_x, _scale, _bias], _y, f"input_0.group_normalization({num_groups}, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

    
        group_normalization_fp8x23_4D()    
        group_normalization_fp8x23_4D_epsilon()    
        group_normalization_fp8x23_4D_groups_equal_to_channels() 
        group_normalization_fp8x23_4D_single_group()    

        group_normalization_fp8x23_3D() 
        group_normalization_fp8x23_3D_epsilon() 
        group_normalization_fp8x23_3D_groups_equal_to_channels()    
        group_normalization_fp8x23_3D_single_group()
        
        group_normalization_fp8x23_2D() 
        group_normalization_fp8x23_2D_epsilon() 
        group_normalization_fp8x23_2D_groups_equal_to_channels()    
        group_normalization_fp8x23_2D_single_group()

        group_normalization_fp8x23_highdim() 
        group_normalization_fp8x23_highdim_epsilon() 
        group_normalization_fp8x23_highdim_groups_equal_to_channels()    
        group_normalization_fp8x23_highdim_single_group()
