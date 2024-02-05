import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Split_to_sequence(RunAll):
    @staticmethod
    def split_to_sequence_u32():
        def split_to_sequence_u32_2d_even_parts():  
            x = np.random.randint(0, 255, (3,6)).astype(np.uint32)
            y= [
                    [
                        np.array([x[0][0:2], x[1][0:2], x[2][0:2]], dtype=np.uint32),
                        np.array([x[0][2:4], x[1][2:4], x[2][2:4]], dtype=np.uint32),
                        np.array([x[0][4:6], x[1][4:6], x[2][4:6]], dtype=np.uint32),
                    ]
                ]
            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.U32, y[0][2].shape, y[0][2].flatten()),
                ]
            
            name = "split_to_sequence_u32_2d_even_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![2].span())), 1, Option::None(()))", name)
            
        def split_to_sequence_u32_2d_odd_parts():  
            x = np.random.randint(0, 255, (3,6)).astype(np.uint32)
            y= [
                    [
                        np.array([x[0][0:5], x[1][0:5], x[2][0:5]], dtype=np.uint32),
                        np.array([x[0][5:6], x[1][5:6], x[2][5:6]], dtype=np.uint32),
                    ]
                ]
            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                ]
            
            name = "split_to_sequence_u32_2d_odd_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![5].span())), 1, Option::None(()))", name)
    
        def split_to_sequence_u32_2d_variable_parts():
            x = np.random.randint(0, 255, (3,6)).astype(np.uint32)
            y= [
                [
                    np.array([x[0][0:]], dtype=np.uint32),
                    np.array([x[1][0:], x[2][0:]], dtype=np.uint32),
                ]
            ]
                        
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                ]
            
            name = "split_to_sequence_u32_2d_variable_parts"
                                                       
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![2].span(), data: array![1,2].span())), 0, Option::None(()))", name)

        def split_to_sequence_u32_2d_nokeepdims():
            x = np.random.randint(0, 255, (3,6)).astype(np.uint32)
            y= [
                [
                    np.array([x[0][0], x[1][0], x[2][0] ], dtype=np.uint32),
                    np.array([x[0][1], x[1][1], x[2][1] ], dtype=np.uint32),
                    np.array([x[0][2], x[1][2], x[2][2] ], dtype=np.uint32),
                    np.array([x[0][3], x[1][3], x[2][3] ], dtype=np.uint32),
                    np.array([x[0][4], x[1][4], x[2][4] ], dtype=np.uint32),
                    np.array([x[0][5], x[1][5], x[2][5] ], dtype=np.uint32),
                ]
            ]
                        
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.U32, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.U32, y[0][3].shape, y[0][3].flatten()),
                    Tensor(Dtype.U32, y[0][4].shape, y[0][4].flatten()),
                    Tensor(Dtype.U32, y[0][5].shape, y[0][5].flatten()),
                ]
            
            name = "split_to_sequence_u32_2d_nokeepdims"
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::None(()), 1, Option::Some((false)))", name)
            
        def split_to_sequence_u32_1d_even_parts():  
            x = np.random.randint(0, 255, (18)).astype(np.uint32)
            y= [
                    [
                        np.array([x[0],x[1]], dtype=np.uint32),
                        np.array([x[2],x[3]], dtype=np.uint32),
                        np.array([x[4],x[5]], dtype=np.uint32),
                        np.array([x[6],x[7]], dtype=np.uint32),
                        np.array([x[8],x[9]], dtype=np.uint32),
                        np.array([x[10],x[11]], dtype=np.uint32),
                        np.array([x[12],x[13]], dtype=np.uint32),
                        np.array([x[14],x[15]], dtype=np.uint32),
                        np.array([x[16],x[17]], dtype=np.uint32),
                      
                    ]
                ]
                            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.U32, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.U32, y[0][3].shape, y[0][3].flatten()),
                    Tensor(Dtype.U32, y[0][4].shape, y[0][4].flatten()),
                    Tensor(Dtype.U32, y[0][5].shape, y[0][5].flatten()),
                    Tensor(Dtype.U32, y[0][6].shape, y[0][6].flatten()),
                    Tensor(Dtype.U32, y[0][7].shape, y[0][7].flatten()),
                    Tensor(Dtype.U32, y[0][8].shape, y[0][8].flatten()),
                ]
            
            name = "split_to_sequence_u32_1d_even_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![2].span())), 0, Option::None(()))", name)


        def split_to_sequence_u32_1d_variable_parts():  
            x = np.random.randint(0, 255, (18)).astype(np.uint32)
            y= [
                [
                    np.array([x[0],x[1], x[2], x[3], x[4]], dtype=np.uint32),
                    np.array([x[5],x[6], x[7]], dtype=np.uint32),
            
            
                ]
            ]
                            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
            
                ]
            
            name = "split_to_sequence_u32_1d_variable_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![2].span(), data: array![5,3].span())), 0, Option::None(()))", name)
            
        def split_to_sequence_u32_1d_nokeepdims():
            x = np.random.randint(0, 255, (18)).astype(np.uint32)
            y= [
                [
                    np.array(x[0], dtype=np.uint32),
                    np.array(x[1], dtype=np.uint32),
                    np.array(x[2], dtype=np.uint32),
                    np.array(x[3], dtype=np.uint32),
                    np.array(x[4], dtype=np.uint32),
                    np.array(x[5], dtype=np.uint32),
                    np.array(x[6], dtype=np.uint32),
                    np.array(x[7], dtype=np.uint32),
                    np.array(x[8], dtype=np.uint32),
                    np.array(x[9], dtype=np.uint32),
                    np.array(x[10], dtype=np.uint32),
                    np.array(x[11], dtype=np.uint32),
                    np.array(x[12], dtype=np.uint32),
                    np.array(x[13], dtype=np.uint32),
                    np.array(x[14], dtype=np.uint32),
                    np.array(x[15], dtype=np.uint32),
                    np.array(x[16], dtype=np.uint32),
                    np.array(x[17], dtype=np.uint32),
                ]
            ]
                        
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.U32, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.U32, y[0][3].shape, y[0][3].flatten()),
                    Tensor(Dtype.U32, y[0][4].shape, y[0][4].flatten()),
                    Tensor(Dtype.U32, y[0][5].shape, y[0][5].flatten()),
                    Tensor(Dtype.U32, y[0][6].shape, y[0][6].flatten()),
                    Tensor(Dtype.U32, y[0][7].shape, y[0][7].flatten()),
                    Tensor(Dtype.U32, y[0][8].shape, y[0][8].flatten()),
                    Tensor(Dtype.U32, y[0][9].shape, y[0][9].flatten()),
                    Tensor(Dtype.U32, y[0][10].shape, y[0][10].flatten()),
                    Tensor(Dtype.U32, y[0][11].shape, y[0][11].flatten()),
                    Tensor(Dtype.U32, y[0][12].shape, y[0][12].flatten()),
                    Tensor(Dtype.U32, y[0][13].shape, y[0][13].flatten()),
                    Tensor(Dtype.U32, y[0][14].shape, y[0][14].flatten()),
                    Tensor(Dtype.U32, y[0][15].shape, y[0][15].flatten()),
                    Tensor(Dtype.U32, y[0][16].shape, y[0][16].flatten()),
                    Tensor(Dtype.U32, y[0][17].shape, y[0][17].flatten()),
                ]
            
            name = "split_to_sequence_u32_1d_nokeepdims"
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::None(()), 0, Option::Some((false)))", name)

        def split_to_sequence_u32_1d_odd_parts():  
            x = np.random.randint(0, 255, (18)).astype(np.uint32)
            y= [
                    [
                        np.array(x[0:5], dtype=np.uint32),
                        np.array(x[5:10], dtype=np.uint32),
                        np.array(x[10:15], dtype=np.uint32),
                        np.array(x[15:], dtype=np.uint32),
                
                    ]
                ]
            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.U32, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.U32, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.U32, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.U32, y[0][3].shape, y[0][3].flatten()),
                ]
            
            name = "split_to_sequence_u32_1d_odd_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![5].span())), 0, Option::None(()))", name)
            
            
        split_to_sequence_u32_2d_even_parts()    #along axis 1
        split_to_sequence_u32_2d_odd_parts()     #along axis 1
        split_to_sequence_u32_2d_variable_parts() #along axis 0
        split_to_sequence_u32_2d_nokeepdims()     #along axis 1
        split_to_sequence_u32_1d_even_parts()     #along axis 0
        split_to_sequence_u32_1d_variable_parts() #along axis 0
        split_to_sequence_u32_1d_nokeepdims()     #along axis 0
        split_to_sequence_u32_1d_odd_parts()      #along axis 0

    
    
    @staticmethod 
    def split_to_sequence_fp16x16():
        def split_to_sequence_fp16x16_2d_even_parts():  
            x = to_fp(np.random.randint(-125, 125, (3, 6)).astype(np.int64), FixedImpl.FP16x16)

            y = [
                    [
                        np.array([x[0][0:2],x[1][0:2], x[2][0:2]] ).astype(np.int64),
                        np.array([x[0][2:4],x[1][2:4], x[2][2:4]] ).astype(np.int64),
                        np.array([x[0][4:6],x[1][4:6], x[2][4:6]] ).astype(np.int64),
                    ]
                ]

            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.FP16x16, y[0][2].shape, (y[0][2].flatten()))
                ]
            
            name = "split_to_sequence_fp16x16_2d_even_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![2].span())), 1, Option::None(()))", name)

        def split_to_sequence_fp16x16_1d_even_parts():  
            x = to_fp(np.random.randint(-125, 125, (18)).astype(np.int64), FixedImpl.FP16x16)

            y = [
                        [
                            np.array(x[0:2] ).astype(np.int64),
                            np.array(x[2:4] ).astype(np.int64),
                            np.array(x[4:6] ).astype(np.int64),
                            np.array(x[6:8] ).astype(np.int64),
                            np.array(x[8:10] ).astype(np.int64),
                            np.array(x[10:12] ).astype(np.int64),
                            np.array(x[12:14] ).astype(np.int64),
                            np.array(x[14:16] ).astype(np.int64),
                            np.array(x[16:18] ).astype(np.int64),
                    
                        ]
                    ]

            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.FP16x16, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.FP16x16, y[0][3].shape, y[0][3].flatten()),
                    Tensor(Dtype.FP16x16, y[0][4].shape, y[0][4].flatten()),
                    Tensor(Dtype.FP16x16, y[0][5].shape, y[0][5].flatten()),
                    Tensor(Dtype.FP16x16, y[0][6].shape, y[0][6].flatten()),
                    Tensor(Dtype.FP16x16, y[0][7].shape, y[0][7].flatten()),
                    Tensor(Dtype.FP16x16, y[0][8].shape, y[0][8].flatten())
                ]
            
            name = "split_to_sequence_fp16x16_1d_even_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![2].span())), 0, Option::None(()))", name)
    
        def split_to_sequence_fp16x16_2d_variable_parts():
            x = to_fp(np.random.randint(-125, 125, (3, 6)).astype(np.int64), FixedImpl.FP16x16)

            y = [
                    [
                        np.array([x[0][0:]], dtype=np.int64),
                        np.array([x[1][0:], x[2][0:]], dtype=np.int64),
                    ]
                ]
            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten())
                ]
            
            name = "split_to_sequence_fp16x16_2d_variable_parts"
                                                       
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![2].span(), data: array![1,2].span())), 0, Option::None(()))", name)

        def split_to_sequence_fp16x16_2d_odd_parts():  
            x = to_fp(np.random.randint(-125, 125, (3, 6)).astype(np.int64), FixedImpl.FP16x16)
            y= [
                    [
                        np.array([x[0][0:5], x[1][0:5], x[2][0:5]], dtype=np.int64),
                        np.array([x[0][5:6], x[1][5:6], x[2][5:6]], dtype=np.int64),
                    ]
                ]
            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape,  y[0][1].flatten())
                ]
            
            name = "split_to_sequence_fp16x16_2d_odd_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![5].span())), 1, Option::None(()))", name)

        def split_to_sequence_fp16x16_2d_nokeepdims():
            x = to_fp(np.random.randint(-125, 125, (3, 6)).astype(np.int64), FixedImpl.FP16x16)

            y= [
                        [
                            np.array([x[0][0], x[1][0], x[2][0] ], dtype=np.int64),
                            np.array([x[0][1], x[1][1], x[2][1] ], dtype=np.int64),
                            np.array([x[0][2], x[1][2], x[2][2] ], dtype=np.int64),
                            np.array([x[0][3], x[1][3], x[2][3] ], dtype=np.int64),
                            np.array([x[0][4], x[1][4], x[2][4] ], dtype=np.int64),
                            np.array([x[0][5], x[1][5], x[2][5] ], dtype=np.int64),
                        ]
                    ]
            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.FP16x16, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.FP16x16, y[0][3].shape, y[0][3].flatten()),
                    Tensor(Dtype.FP16x16, y[0][4].shape, y[0][4].flatten()),
                    Tensor(Dtype.FP16x16, y[0][5].shape, y[0][5].flatten())

                ]
            
            name = "split_to_sequence_fp16x16_2d_nokeepdims"
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::None(()), 1, Option::Some((false)))", name)
            
        def split_to_sequence_fp16x16_1d_variable_parts():   
            x = to_fp(np.random.randint(-125, 125, (18)).astype(np.int64), FixedImpl.FP16x16)
            y= [
                [
                    np.array([x[0],x[1], x[2], x[3], x[4]], dtype=np.int64),
                    np.array([x[5],x[6], x[7]], dtype=np.int64),
            
            
                ]
            ]
                            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten())
                ]
            
            name = "split_to_sequence_fp16x16_1d_variable_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![2].span(), data: array![5,3].span())), 0, Option::None(()))", name)

        def split_to_sequence_fp16x16_1d_odd_parts():  
            x = to_fp(np.random.randint(-125, 125, (18)).astype(np.int64), FixedImpl.FP16x16)
            y= [
                    [
                        np.array(x[0:5], dtype=np.int64),
                        np.array(x[5:10], dtype=np.int64),
                        np.array(x[10:15], dtype=np.int64),
                        np.array(x[15:], dtype=np.int64),
                
                    ]
                ]
            
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                    Tensor(Dtype.FP16x16, y[0][0].shape, y[0][0].flatten()),
                    Tensor(Dtype.FP16x16, y[0][1].shape, y[0][1].flatten()),
                    Tensor(Dtype.FP16x16, y[0][2].shape, y[0][2].flatten()),
                    Tensor(Dtype.FP16x16, y[0][3].shape, y[0][3].flatten())
                ]
            
            name = "split_to_sequence_fp16x16_1d_odd_parts" 
            
            make_test([_x], _y, "input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![].span(), data: array![5].span())), 0, Option::None(()))", name)
            

    

        split_to_sequence_fp16x16_2d_even_parts()    #along axis 1
        split_to_sequence_fp16x16_1d_even_parts()     #along axis 0
        split_to_sequence_fp16x16_2d_variable_parts() #along axis 0
        split_to_sequence_fp16x16_2d_odd_parts()     #along axis 1
        split_to_sequence_fp16x16_2d_nokeepdims()     #along axis 1
        split_to_sequence_fp16x16_1d_variable_parts() #along axis 0
        split_to_sequence_fp16x16_1d_odd_parts()      #along axis 0


    