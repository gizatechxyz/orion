#[derive(Copy, Drop)]
enum AUTO_PAD {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID
}

#[derive(Copy, Drop)]
enum POOLING_TYPE {
    AVG,
    LPPOOL,
    MAX,
}

#[derive(Copy, Drop)]
enum MODE {
    NEAREST,
    LINEAR,
    CUBIC,
}

#[derive(Copy, Drop)]
enum PADDING_MODE {
    ZEROS,
    BORDER,
    REFLECTION,
}
