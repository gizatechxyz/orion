import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

class Instance_normalization(RunAll):
    @staticmethod
    def instance_normalization_fp16x16():
        def instance_normalization_fp16x16_4D():
            b = 1
            c = 2
            h = 1
            w = 3
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_4D"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)

        def instance_normalization_fp16x16_4D_epsilon(): 
            b = 2
            c = 3
            h = 4
            w = 5
            epsilon = 1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon ).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_4D_epsilon"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 ,  Option::Some( FixedTrait::new(6554, false)) )", name)


        def instance_normalization_fp16x16_4D_single_batch(): 
            b = 1
            c = 3
            h = 4
            w = 5
            epsilon = 1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon ).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_4D_single_batch"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 ,  Option::Some( FixedTrait::new(6554, false)) )", name)


        def instance_normalization_fp16x16_4D_eq_batch_channel(): 
            b = 2
            c = 2
            h = 3
            w = 4
            epsilon = 1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon ).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_4D_eq_batch_channel"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 ,  Option::Some( FixedTrait::new(6554, false)) )", name)

        
        def instance_normalization_fp16x16_3D(): 
            b = 4
            c = 2
            n1 = 5
            x = np.random.randn(b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_3D"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)


        def instance_normalization_fp16x16_3D_epsilon(): 
            b = 2
            c = 5
            n1 = 5
            epsilon = 1e-1
            x = np.random.randn(b, c, n1,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_3D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)
        
        
        def instance_normalization_fp16x16_3D_single_batch(): 
            b = 1
            c = 5
            n1 = 4
            x = np.random.randn(b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_3D_single_batch"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)


        def instance_normalization_fp16x16_3D_eq_batch_channel(): 
            b = 2
            c = 2
            n1 = 4
            x = np.random.randn(b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_3D_eq_batch_channel"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)
            


      
        def instance_normalization_fp16x16_2D(): 
            b = 4
            c = 5
            x = np.random.randn(b, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_2D"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)



        def instance_normalization_fp16x16_2D_epsilon(): 
            b = 3
            c = 5
            epsilon =  1e-1
            x = np.random.randn(b, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)

        
        def instance_normalization_fp16x16_2D_single_batch(): 
            b = 1
            c = 5
            epsilon =  1e-1
            x = np.random.randn(b, c, ).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_2D_single_batch"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)


        def instance_normalization_fp16x16_2D_eq_batch_channel(): 
            b = 2
            c = 2
            n1 = 4
            x = np.random.randn(b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
            
            name = "instance_normalization_fp16x16_2D_eq_batch_channel"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)


        def instance_normalization_fp16x16_highdim(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,5)
            c = 3
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "instance_normalization_fp16x16_highdim"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)


        def instance_normalization_fp16x16_highdim_epsilon(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,5)
            c = 3
            epsilon =  1e-1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "instance_normalization_fp16x16_highdim_epsilon"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) )", name)
    
        
        def instance_normalization_fp16x16_highdim_batch_eq_channels(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = 2
            c = 2
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "instance_normalization_fp16x16_highdim_batch_eq_channels"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)
    
    
        def instance_normalization_fp16x16_highdim_single_batch(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = 1
            c = 2
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
            _scale = Tensor(Dtype.FP16x16, scale.shape, to_fp(scale.flatten(), FixedImpl.FP16x16))
            _bias =  Tensor(Dtype.FP16x16, bias.shape, to_fp(bias.flatten(), FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16));
            
            name = "instance_normalization_fp16x16_highdim_single_batch"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)
        
        instance_normalization_fp16x16_4D()    
        instance_normalization_fp16x16_4D_epsilon()
        instance_normalization_fp16x16_4D_single_batch()
        instance_normalization_fp16x16_4D_eq_batch_channel()
        
        instance_normalization_fp16x16_3D() 
        instance_normalization_fp16x16_3D_epsilon()   
        instance_normalization_fp16x16_3D_single_batch()
        instance_normalization_fp16x16_3D_eq_batch_channel()
        
        instance_normalization_fp16x16_2D() 
        instance_normalization_fp16x16_2D_epsilon()  
        instance_normalization_fp16x16_2D_single_batch()
        instance_normalization_fp16x16_2D_eq_batch_channel()

        instance_normalization_fp16x16_highdim()
        instance_normalization_fp16x16_highdim_epsilon()
        instance_normalization_fp16x16_highdim_batch_eq_channels()
        instance_normalization_fp16x16_highdim_single_batch()
        
   



    @staticmethod
    def instance_normalization_fp8x23():
        def instance_normalization_fp8x23_4D():
            b = 1
            c = 2
            h = 1
            w = 3
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_4D"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()))", name)
    
        def instance_normalization_fp8x23_4D_epsilon(): 
            b = 2
            c = 3
            h = 4
            w = 5
            epsilon =  1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_4D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)



        def instance_normalization_fp8x23_4D_single_batch(): 
            b = 1
            c = 3
            h = 4
            w = 5
            epsilon =  1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_4D_single_batch"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

        def instance_normalization_fp8x23_4D_eq_batch_channel(): 
            b = 2
            c = 2
            h = 3
            w = 4
            epsilon =  1e-1
            x = np.random.randn(b, c, h, w).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_4D_eq_batch_channel"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

   
        def instance_normalization_fp8x23_3D():
            b = 2
            c = 4
            n1 = 5
            x = np.random.randn( b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_3D"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()))", name)
    
    
        def instance_normalization_fp8x23_3D_epsilon():
            b = 4
            c = 3
            n1 = 5
            epsilon = 1e-1
            x = np.random.randn( b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_3D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    

        def instance_normalization_fp8x23_3D_single_batch():
            b = 1
            c = 4
            n1 = 5
            x = np.random.randn( b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_3D_single_batch"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()))", name)
            

        def instance_normalization_fp8x23_3D_eq_batch_channel(): 
            b = 2
            c = 2
            n1 = 4
            epsilon =  1e-1
            x = np.random.randn( b, c, n1).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_3D_eq_batch_channel"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)


        def instance_normalization_fp8x23_2D():
            b = 5
            c = 4
            x = np.random.randn( b, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()))) )", name)


        def instance_normalization_fp8x23_2D_epsilon():
            b = 5
            c = 5
            epsilon = 1e-1
            x = np.random.randn( b, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    


        def instance_normalization_fp8x23_2D_single_batch():
            b = 1
            c = 5
            epsilon = 1e-1
            x = np.random.randn( b, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_2D_epsilon"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)

        def instance_normalization_fp8x23_2D_eq_batch_channel(): 
            b = 2
            c = 2
            epsilon =  1e-1
            x = np.random.randn( b, c,).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_2D_eq_batch_channel"
            make_test([_x, _scale, _bias,], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)


        def instance_normalization_fp8x23_highdim(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,5)
            c = 3
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_highdim"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)


        def instance_normalization_fp8x23_highdim_epsilon(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = np.random.randint(1,5)
            c = 3
            epsilon =  1e-1
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias, epsilon).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_highdim_epsilon"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) )", name)
    
        
        def instance_normalization_fp8x23_highdim_batch_eq_channels(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = 2
            c = 2
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_highdim_batch_eq_channels"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)
    
    
        def instance_normalization_fp8x23_highdim_single_batch(): 
            extra_dims =[ np.random.randint(1,4) for i in range(1, np.random.randint(4,7))]
            b = 1
            c = 2
            x = np.random.randn( b, c, *extra_dims).astype(np.float32)
            scale = np.random.randn(c).astype(np.float32)
            bias = np.random.randn(c).astype(np.float32)
            y = _instancenorm_test_mode(x, scale, bias).astype(np.float32)
            
            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
            _scale = Tensor(Dtype.FP8x23, scale.shape, to_fp(scale.flatten(), FixedImpl.FP8x23))
            _bias =  Tensor(Dtype.FP8x23, bias.shape, to_fp(bias.flatten(), FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
            
            name = "instance_normalization_fp8x23_highdim_single_batch"
            make_test([_x, _scale, _bias], _y, f"input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) )", name)
    
        instance_normalization_fp8x23_4D()    
        instance_normalization_fp8x23_4D_epsilon()
        instance_normalization_fp8x23_4D_single_batch()
        instance_normalization_fp8x23_4D_eq_batch_channel()
        
        instance_normalization_fp8x23_3D() 
        instance_normalization_fp8x23_3D_epsilon()   
        instance_normalization_fp8x23_3D_single_batch()
        instance_normalization_fp8x23_3D_eq_batch_channel()
        
        instance_normalization_fp8x23_2D() 
        instance_normalization_fp8x23_2D_epsilon()  
        instance_normalization_fp8x23_2D_single_batch() 
        instance_normalization_fp8x23_2D_eq_batch_channel()

        instance_normalization_fp8x23_highdim()
        instance_normalization_fp8x23_highdim_epsilon()
        instance_normalization_fp8x23_highdim_batch_eq_channels()
        instance_normalization_fp8x23_highdim_single_batch()

