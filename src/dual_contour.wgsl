[[group(0), binding(0)]]
var sdf: [[access(read)]] texture_storage_3d<r32float>;

[[group(0), binding(1)]]
var surface: [[access(write)]] texture_storage_3d<r32uint>;

fn check_surface(cell_min: vec3<u32>) -> u32 {
    var num_negative: u32 = 0u;
    for (var dx: u32 = 0u; dx <= 1u; dx = dx + 1u) {
        for (var dy: u32 = 0u; dy <= 1u; dy = dy + 1u) {
            for (var dz: u32 = 0u; dz <= 1u; dz = dz + 1u) {
                const corner: vec3<u32> = cell_min + vec3<u32>(dx, dy, dz);
                const corner_d: f32 = textureLoad(sdf, vec3<i32>(corner)).r;
                if (corner_d < 0.0) {
                    num_negative = num_negative + 1u;
                }
            }
        }
    }
    if (num_negative != 0u && num_negative != 8u) {
        return 1u;
    } else {
        return 0u;
    }
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    textureStore(surface, vec3<i32>(global_id), vec4<u32>(check_surface(global_id), 0u, 0u, 1u));
}
