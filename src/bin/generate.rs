use wgpu_isosurface::DualContourPipeline;

use futures::{executor::block_on, Future};
use std::sync::Arc;

fn main() {
    // WGPU initialization.
    let adapter = block_on(get_default_adapter()).unwrap();
    println!("Using adapter: {:?}", adapter);
    let (device, queue) = block_on(request_default_device(&adapter));

    let device = Arc::new(device);

    spawn_polling_thread(device.clone());

    let shader_flags = default_shader_flags(adapter.get_info().backend);

    // Actually run the compute shader.
    let pipe = DualContourPipeline::new(&device, shader_flags);

    for _ in 0..10 {
        timed_dispatch(&pipe, &device, &queue);
    }
}

fn timed_dispatch(pipe: &DualContourPipeline, device: &wgpu::Device, queue: &wgpu::Queue) {
    let input = vec![1000; 10000];

    let t1 = std::time::Instant::now();
    let output = pipe.dispatch(&input, &device, &queue);

    let t2 = std::time::Instant::now();
    let _output = block_on(output.unwrap());
    let t3 = std::time::Instant::now();

    let d1 = (t2 - t1).as_micros();
    let d2 = (t3 - t2).as_micros();

    println!("Dipatch took {} micros", d1);
    println!("Unwrap took {} micros", d2);
}

fn spawn_polling_thread(device: Arc<wgpu::Device>) {
    std::thread::spawn(move || loop {
        device.poll(wgpu::Maintain::Wait);
    });
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
