mod core;
mod implementations;
mod functional;

use orion::operators::sequence::core::SequenceTrait;

use orion::operators::sequence::implementations::sequence_fp8x23::FP8x23Sequence;
use orion::operators::sequence::implementations::sequence_fp8x23wide::FP8x23WSequence;
use orion::operators::sequence::implementations::sequence_fp16x16::FP16x16Sequence;
use orion::operators::sequence::implementations::sequence_fp16x16wide::FP16x16WSequence;
use orion::operators::sequence::implementations::sequence_i8::I8Sequence;
use orion::operators::sequence::implementations::sequence_i32::I32Sequence;
use orion::operators::sequence::implementations::sequence_u32::U32Sequence;
use orion::operators::sequence::implementations::sequence_bool::BoolSequence;
