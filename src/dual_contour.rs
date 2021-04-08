use futures::Future;
use std::borrow::Cow;
use std::convert::TryInto;
use wgpu::util::DeviceExt;

pub struct DualContourPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl DualContourPipeline {
    pub fn new(device: &wgpu::Device, shader_flags: wgpu::ShaderFlags) -> Self {
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

        Self { pipeline }
    }

    pub async fn dispatch(
        &self,
        input: &[u32],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> DualContourOutputBuffer {
        // GPU-side buffer for shader input and output.
        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // Bind the storage to the shader interface.
            let bind_group_layout = self.pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                }],
            });

            // Encode our commands to dispatch the pipeline.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.insert_debug_marker("compute collatz iterations");
            pass.dispatch(input.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }

        // Encode a command to copy the output back to our staging buffer.
        let buffer_size_bytes = std::mem::size_of_val(input) as wgpu::BufferAddress;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size_bytes,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, buffer_size_bytes);

        // Submits command encoder for processing.
        queue.submit(Some(encoder.finish()));

        // Gets the future representing when `staging_buffer` can be read from
        let buffer_is_mapped = Box::new(staging_buffer.slice(..).map_async(wgpu::MapMode::Read));

        DualContourOutputBuffer {
            staging_buffer,
            buffer_is_mapped,
        }
    }
}

pub struct DualContourOutputBuffer {
    staging_buffer: wgpu::Buffer,
    buffer_is_mapped: Box<dyn Future<Output = Result<(), wgpu::BufferAsyncError>> + Unpin>,
}

impl DualContourOutputBuffer {
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
