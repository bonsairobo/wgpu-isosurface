use futures::Future;
use std::borrow::Cow;
use std::convert::TryInto;

pub struct DualContourPipeline {
    pipeline: wgpu::ComputePipeline,
    buffer_size_bytes: u64,
    staging_in_buffer: wgpu::Buffer,
    gpu_in_buffer: wgpu::Buffer,
    gpu_out_buffer: wgpu::Buffer,
    staging_out_buffer: wgpu::Buffer,
}

impl DualContourPipeline {
    pub fn new(
        device: &wgpu::Device,
        shader_flags: wgpu::ShaderFlags,
        buffer_size_bytes: wgpu::BufferAddress,
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

        let staging_in_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Staging Buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        let gpu_in_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Storage Buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::STORAGE,
            mapped_at_creation: false,
        });
        let gpu_out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Storage Buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            buffer_size_bytes,
            staging_in_buffer,
            gpu_in_buffer,
            gpu_out_buffer,
            staging_out_buffer,
        }
    }

    pub async fn dispatch<'a>(
        &'a self,
        input: &[u32],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> DualContourOutputBuffer<'a> {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let in_slice = self.staging_in_buffer.slice(..);
        in_slice.map_async(wgpu::MapMode::Write).await.unwrap();
        in_slice
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(input));
        self.staging_in_buffer.unmap();

        encoder.copy_buffer_to_buffer(
            &self.staging_in_buffer,
            0,
            &self.gpu_in_buffer,
            0,
            self.buffer_size_bytes,
        );

        {
            // Bind the storage to the shader interface.
            let bind_group_layout = self.pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.gpu_in_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.gpu_out_buffer.as_entire_binding(),
                    },
                ],
            });

            // Encode our commands to dispatch the pipeline.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // pass.insert_debug_marker("compute collatz iterations");
            pass.dispatch(input.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }

        // Encode a command to copy the output back to our staging buffer.
        encoder.copy_buffer_to_buffer(
            &self.gpu_out_buffer,
            0,
            &self.staging_out_buffer,
            0,
            self.buffer_size_bytes,
        );

        // Submits command encoder for processing.
        queue.submit(Some(encoder.finish()));

        // Gets the future representing when `staging_buffer` can be read from
        let buffer_is_mapped = Box::new(
            self.staging_out_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read),
        );

        DualContourOutputBuffer {
            staging_buffer: &self.staging_out_buffer,
            buffer_is_mapped,
        }
    }
}

pub struct DualContourOutputBuffer<'a> {
    staging_buffer: &'a wgpu::Buffer,
    buffer_is_mapped: Box<dyn Future<Output = Result<(), wgpu::BufferAsyncError>> + Unpin>,
}

impl<'a> DualContourOutputBuffer<'a> {
    pub async fn unwrap(self) -> Vec<u32> {
        // Awaits until `staging_buffer` can be read from.
        match self.buffer_is_mapped.await {
            Ok(()) => {
                let result = {
                    let data = self.staging_buffer.slice(..).get_mapped_range();

                    // Converts bytes back to u32.
                    data.chunks_exact(4)
                        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                        .collect()
                }; // With the current interface, we have to make sure all mapped views are dropped before we unmap the buffer.
                self.staging_buffer.unmap();

                result
            }
            Err(e) => {
                panic!("Failed to unwrap DualContourOutputBuffer: {:?}", e)
            }
        }
    }
}
