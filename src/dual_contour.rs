use crate::{aligned_texture_buffer_size_bytes, aligned_texture_bytes_per_row, extent_volume};

use futures::Future;
use std::borrow::Cow;
use std::convert::TryInto;
use std::num::NonZeroU32;

pub struct DualContourPipeline {
    pipeline: wgpu::ComputePipeline,
    sdf_texture: wgpu::Texture,
    surface_texture: wgpu::Texture,
    surface_staging_buffer: wgpu::Buffer,
    extent: wgpu::Extent3d,
}

impl DualContourPipeline {
    pub fn new(
        device: &wgpu::Device,
        shader_flags: wgpu::ShaderFlags,
        extent: wgpu::Extent3d,
    ) -> Self {
        let module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("dual_contour.wgsl"))),
            flags: shader_flags,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
        });

        let sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::STORAGE,
        });
        let surface_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surface Texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsage::COPY_SRC | wgpu::TextureUsage::STORAGE,
        });

        let surface_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Staging Buffer"),
            size: aligned_texture_buffer_size_bytes::<u32>(extent),
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            sdf_texture,
            surface_texture,
            surface_staging_buffer,
            extent,
        }
    }

    pub async fn dispatch<'a>(
        &'a self,
        sdf: &[f32],
        sdf_dimensions: [u32; 3],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> DualContourOutputBuffer<'a> {
        assert_eq!(sdf.len() as u64, extent_volume(self.extent));

        // This takes care of buffer alignment / padding for us, so it's simpler than copy_buffer_to_texture.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.sdf_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            bytemuck::cast_slice(sdf),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(self.extent.width * 4).unwrap()),
                rows_per_image: Some(NonZeroU32::new(self.extent.height).unwrap()),
            },
            self.extent,
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let sdf_texture_view = self
            .sdf_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let surface_texture_view = self
            .surface_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Bind the storage to the shader interface.
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&sdf_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&surface_texture_view),
                },
            ],
        });

        {
            // Encode our commands to dispatch the pipeline.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // The compute kernel is 2x2x2, so don't look at all texels.
            pass.dispatch(
                sdf_dimensions[0] - 1,
                sdf_dimensions[1] - 1,
                sdf_dimensions[2] - 1,
            );
        }

        // Encode a command to copy the output back to our staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.surface_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.surface_staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        NonZeroU32::new(aligned_texture_bytes_per_row::<u32>(self.extent.width))
                            .unwrap(),
                    ),
                    rows_per_image: Some(NonZeroU32::new(self.extent.height).unwrap()),
                },
            },
            self.extent,
        );

        // Submits command encoder for processing.
        queue.submit(Some(encoder.finish()));

        // Gets the future representing when `buffer` can be read from
        let buffer_is_mapped = Box::new(
            self.surface_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read),
        );

        DualContourOutputBuffer {
            buffer: &self.surface_staging_buffer,
            buffer_is_mapped,
        }
    }
}

pub struct DualContourOutputBuffer<'a> {
    buffer: &'a wgpu::Buffer,
    buffer_is_mapped: Box<dyn Future<Output = Result<(), wgpu::BufferAsyncError>> + Unpin>,
}

impl<'a> DualContourOutputBuffer<'a> {
    pub async fn unwrap(self) -> Vec<u32> {
        // Awaits until `buffer` can be read from.
        match self.buffer_is_mapped.await {
            Ok(()) => {
                let result = {
                    let data = self.buffer.slice(..).get_mapped_range();

                    // TODO: filter out the alignment padding
                    // Converts bytes back to u32.
                    data.chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect()
                }; // With the current interface, we have to make sure all mapped views are dropped before we unmap the buffer.
                self.buffer.unmap();

                result
            }
            Err(e) => {
                panic!("Failed to unwrap DualContourOutputBuffer: {:?}", e)
            }
        }
    }
}
