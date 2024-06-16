use crate::cairograph::{CairoGraph, CairoNode, CairoOps};
use orion::primgraph::{PrimGraph, Primops};

pub fn primgraph_to_cairograph(primgraph: PrimGraph) -> CairoGraph {
    let mut cairo_nodes = vec![];

    for node in &primgraph.nodes {
        let optype = match node.optype {
            Primops::Log2 => CairoOps::Log2,
            Primops::Exp2 => CairoOps::Exp2,
            Primops::Sin => CairoOps::Sin,
            Primops::Sqrt => CairoOps::Sqrt,
            Primops::Recip => CairoOps::Recip,
            Primops::Add => CairoOps::Add,
            Primops::Mul => CairoOps::Mul,
            Primops::Mod => CairoOps::Mod,
            Primops::LessThan => CairoOps::LessThan,
            Primops::SumReduce => CairoOps::SumReduce,
            Primops::MaxReduce => CairoOps::MaxReduce,
            Primops::Reshape => CairoOps::Reshape,
        };
        cairo_nodes.push(CairoNode {
            name: node.name.clone(),
            optype,
            inputs: node.inputs.clone(),
            attributes: node.attributes.clone(),
            outputs: node.outputs.clone(),
        })
    }

    CairoGraph {
        inputs: primgraph.inputs.clone(),
        outputs: primgraph.outputs.clone(),
        nodes: cairo_nodes,
        initializers: primgraph.initializers.clone(),
        shape: primgraph.shape.clone(),
    }
}
