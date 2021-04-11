pub fn aligned_texture_buffer_size_bytes<T>(extent: wgpu::Extent3d) -> u64 {
    let bytes_per_row = aligned_texture_bytes_per_row::<T>(extent.width);

    bytes_per_row as u64 * extent.height as u64 * extent.depth_or_array_layers as u64
}

pub fn aligned_texture_bytes_per_row<T>(width: u32) -> u32 {
    let texel_size = std::mem::size_of::<T>();

    assert!(texel_size > 0);
    assert!(width > 0);

    next_multiple(
        width * texel_size as u32,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    )
}

pub fn next_multiple(x: u32, factor: u32) -> u32 {
    if x % factor != 0 {
        factor * ((x / factor) + 1)
    } else {
        x
    }
}

pub fn extent_volume(e: wgpu::Extent3d) -> u64 {
    (e.width * e.height * e.depth_or_array_layers) as u64
}
