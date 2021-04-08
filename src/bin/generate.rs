use wgpu_isosurface::DualContourPipeline;

use futures::{executor::block_on, Future};

fn main() {
    // WGPU initialization.
    let adapter = block_on(get_default_adapter()).unwrap();
    let (device, queue) = block_on(request_default_device(&adapter));
    let shader_flags = default_shader_flags(adapter.get_info().backend);

    // Actually run the compute shader.
    let shader = DualContourPipeline::new(&device, shader_flags);
    let input = vec![999; 100000];

    let before = std::time::Instant::now();

    let output = block_on(shader.dispatch(&input, &device, &queue));

    // Poll the device in a blocking manner so that our future resolves. In an actual application, `device.poll(...)` should be
    // called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    let _output = block_on(output.unwrap());

    let after = std::time::Instant::now();

    // println!("{:?}", output);

    let duration = (after - before).as_micros();
    println!("Took {} micros", duration);
}

fn default_shader_flags(backend: wgpu::Backend) -> wgpu::ShaderFlags {
    let mut shader_flags = wgpu::ShaderFlags::VALIDATION;
    if let wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Gl = backend {
        shader_flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
    }

    shader_flags
}

fn get_default_adapter() -> impl Future<Output = Option<wgpu::Adapter>> {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    };

    instance.request_adapter(&adapter_options)
}

async fn request_default_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap()
}
