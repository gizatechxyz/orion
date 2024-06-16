use std::vec;

use onnx::onnx_proto::NodeProto;
use orion::helpers::{vec_i32_into_raw_data, vec_u32_into_raw_data};
use orion::primgraph::{Attribute, Dtype, Initializer, PrimNode, Primops, Tensor};

pub(crate) fn decompose_matmul(
    node: NodeProto,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
) -> (
    Vec<PrimNode>,
    Vec<String>,
    Vec<Vec<usize>>,
    Vec<Initializer>,
) {
    if shape_a.len() == 1 && shape_b.len() == 1 {
        // AxA -> A
        assert!(
            *shape_a.first().unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );

        let shape_mul = shape_a.clone();
        let shape_output = vec![*shape_a.first().unwrap()];

        let (nodes, initializer) = matmul(node.clone(), 0, node.input.clone());

        (
            nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
            ],
            vec![shape_output, shape_mul],
            vec![initializer],
        )
    } else if shape_a.len() == 2 && shape_b.len() == 1 {
        // ABxB -> A
        assert!(
            *shape_a.get(1).unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );

        let shape_mul = shape_a.clone();
        let shape_output = vec![*shape_a.first().unwrap()];

        let (nodes, initializer) = matmul(node.clone(), 1, node.input.clone());

        (
            nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
            ],
            vec![shape_output, shape_mul],
            vec![initializer],
        )
    } else if shape_a.len() == 1 && shape_b.len() == 2 {
        // AxAB -> B
        assert!(
            *shape_a.first().unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );

        let shape_mul = shape_b.clone();
        let shape_output = vec![*shape_b.get(1).unwrap()];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (mut nodes, initializer) = matmul(
            node.clone(),
            0,
            vec![
                format!("{}_reshape_a_output", node.name),
                (*node.input.get(1).unwrap()).clone(),
            ],
        );

        let mut vec_nodes = vec![reshape_a];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a],
            vec![initializer, reshape_a_initializer],
        )
    } else if shape_a.len() == 2 && shape_b.len() == 2 {
        // ABxBC -> AC
        assert!(
            *shape_a.get(1).unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );

        let shape_mul = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_b.get(1).unwrap(),
        ];
        let shape_output = vec![*shape_a.first().unwrap(), *shape_b.get(1).unwrap()];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (mut nodes, initializer) = matmul(
            node.clone(),
            1,
            vec![
                format!("{}_reshape_a_output", node.name),
                (*node.input.get(1).unwrap()).clone(),
            ],
        );

        let mut vec_nodes = vec![reshape_a];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a],
            vec![initializer, reshape_a_initializer],
        )
    } else if shape_a.len() == 3 && shape_b.len() == 2 {
        // ABCxCD -> ABD
        assert!(
            *shape_a.get(2).unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );

        let shape_mul = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_a.get(2).unwrap(),
            *shape_b.get(1).unwrap(),
        ];
        let shape_output = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_b.get(1).unwrap(),
        ];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (mut nodes, initializer) = matmul(
            node.clone(),
            2,
            vec![
                format!("{}_reshape_a_output", node.name),
                (*node.input.get(1).unwrap()).clone(),
            ],
        );

        let mut vec_nodes = vec![reshape_a];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a],
            vec![initializer, reshape_a_initializer],
        )
    } else if shape_a.len() == 3 && shape_b.len() == 3 {
        // ABCxACD -> ABD
        assert!(
            *shape_a.first().unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(2).unwrap() == *shape_b.get(1).unwrap(),
            "Incompatible shapes for matmul"
        );

        let mut shape_mul = shape_a.clone();
        shape_mul.push(*shape_b.get(2).unwrap());

        let shape_output = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_b.get(2).unwrap(),
        ];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (reshape_b, reshape_b_initializer, new_shape_b) = reshape_b(node.clone(), shape_b, 1);

        let (mut nodes, reduce_sum_initializer) = matmul(
            node.clone(),
            2,
            vec![
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
        );

        let mut vec_nodes = vec![reshape_a, reshape_b];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a, new_shape_b],
            vec![
                reshape_a_initializer,
                reshape_b_initializer,
                reduce_sum_initializer,
            ],
        )
    } else if shape_a.len() == 4 && shape_b.len() == 4 {
        // ABCDxABDE -> ABCE
        assert!(
            *shape_a.first().unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(1).unwrap() == *shape_b.get(1).unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(3).unwrap() == *shape_b.get(2).unwrap(),
            "Incompatible shapes for matmul"
        );

        let mut shape_mul = shape_a.clone();
        shape_mul.push(*shape_b.get(3).unwrap());

        let shape_output = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_a.get(2).unwrap(),
            *shape_b.get(3).unwrap(),
        ];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (reshape_b, reshape_b_initializer, new_shape_b) = reshape_b(node.clone(), shape_b, 2);

        let (mut nodes, reduce_sum_initializer) = matmul(
            node.clone(),
            3,
            vec![
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
        );

        let mut vec_nodes = vec![reshape_a, reshape_b];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a, new_shape_b],
            vec![
                reshape_a_initializer,
                reshape_b_initializer,
                reduce_sum_initializer,
            ],
        )
    } else if shape_a.len() == 5 && shape_b.len() == 5 {
        // ABCDExABCEF -> ABCDF
        assert!(
            *shape_a.first().unwrap() == *shape_b.first().unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(1).unwrap() == *shape_b.get(1).unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(2).unwrap() == *shape_b.get(2).unwrap(),
            "Incompatible shapes for matmul"
        );
        assert!(
            *shape_a.get(4).unwrap() == *shape_b.get(3).unwrap(),
            "Incompatible shapes for matmul"
        );

        let mut shape_mul = shape_a.clone();
        shape_mul.push(*shape_b.get(4).unwrap());

        let shape_output = vec![
            *shape_a.first().unwrap(),
            *shape_a.get(1).unwrap(),
            *shape_a.get(2).unwrap(),
            *shape_a.get(3).unwrap(),
            *shape_b.get(4).unwrap(),
        ];

        let (reshape_a, reshape_a_initializer, new_shape_a) = reshape_a(node.clone(), shape_a);
        let (reshape_b, reshape_b_initializer, new_shape_b) = reshape_b(node.clone(), shape_b, 3);

        let (mut nodes, reduce_sum_initializer) = matmul(
            node.clone(),
            4,
            vec![
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
        );

        let mut vec_nodes = vec![reshape_a, reshape_b];
        vec_nodes.append(&mut nodes);

        (
            vec_nodes,
            vec![
                (*node.output.first().unwrap()).clone(),
                format!("{}_mul_output", node.name),
                format!("{}_reshape_a_output", node.name),
                format!("{}_reshape_b_output", node.name),
            ],
            vec![shape_output, shape_mul, new_shape_a, new_shape_b],
            vec![
                reshape_a_initializer,
                reshape_b_initializer,
                reduce_sum_initializer,
            ],
        )
    } else {
        panic!("not supported yet")
    }
}

fn matmul(node: NodeProto, reduce_axis: i32, input: Vec<String>) -> (Vec<PrimNode>, Initializer) {
    let mul = PrimNode {
        name: format!("{}_decomposed_mul", node.name),
        optype: Primops::Mul,
        inputs: input,
        outputs: vec![format!("{}_mul_output", node.name)],
        attributes: vec![],
    };

    let reduce_sum_initializer = Initializer {
        name: format!("{}_reduce_sum_initializer", node.name),
        tensor: Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![reduce_axis]),
            dtype: Dtype::I32,
        },
    };

    let reduce_sum = PrimNode {
        name: format!("{}_decomposed_reduce_sum", node.name),
        optype: Primops::SumReduce,
        inputs: vec![
            (*mul.outputs.first().unwrap()).clone(),
            format!("{}_reduce_sum_initializer", node.name),
        ],
        outputs: node.output.clone(),
        attributes: vec![Attribute {
            name: "keepdims".to_string(),
            value: false,
        }],
    };

    (vec![mul, reduce_sum], reduce_sum_initializer)
}

fn reshape_a(node: NodeProto, shape_a: Vec<usize>) -> (PrimNode, Initializer, Vec<usize>) {
    let mut a_new_shape: Vec<u32> = shape_a.iter().map(|&x| x as u32).collect();
    a_new_shape.push(1);

    let reshape_a_initializer = Initializer {
        name: format!("{}_reshape_a_initializer", node.name),
        tensor: Tensor {
            shape: vec![a_new_shape.len()],
            raw_data: vec_u32_into_raw_data(&a_new_shape),
            dtype: Dtype::U32,
        },
    };

    (
        PrimNode {
            name: format!("{}_decomposed_reshape_a", node.name),
            optype: Primops::Reshape,
            inputs: vec![
                node.input.first().unwrap().to_string(),
                reshape_a_initializer.name.clone(),
            ],
            outputs: vec![format!("{}_reshape_a_output", node.name)],
            attributes: vec![],
        },
        reshape_a_initializer,
        a_new_shape.iter().map(|&x| x as usize).collect(),
    )
}

fn reshape_b(
    node: NodeProto,
    shape_b: Vec<usize>,
    index: usize,
) -> (PrimNode, Initializer, Vec<usize>) {
    let mut b_new_shape: Vec<u32> = shape_b.iter().map(|&x| x as u32).collect();
    b_new_shape.insert(index, 1);

    let reshape_b_initializer = Initializer {
        name: format!("{}_reshape_b_initializer", node.name),
        tensor: Tensor {
            shape: vec![b_new_shape.len()],
            raw_data: vec_u32_into_raw_data(&b_new_shape),
            dtype: Dtype::U32,
        },
    };

    (
        PrimNode {
            name: format!("{}_decomposed_reshape_b", node.name),
            optype: Primops::Reshape,
            inputs: vec![
                node.input.get(1).unwrap().to_string(),
                reshape_b_initializer.name.clone(),
            ],
            outputs: vec![format!("{}_reshape_b_output", node.name)],
            attributes: vec![],
        },
        reshape_b_initializer,
        b_new_shape.iter().map(|&x| x as usize).collect(),
    )
}
