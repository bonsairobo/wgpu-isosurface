[[group(0), binding(0)]]
var sdf: [[access(read)]] texture_storage_3d<r32float>;

[[group(0), binding(1)]]
var surface: [[access(write)]] texture_storage_3d<r32uint>;

var CORNERS: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0),
    vec3<i32>(0, 0, 1),
    vec3<i32>(0, 1, 0),
    vec3<i32>(0, 1, 1),
    vec3<i32>(1, 0, 0),
    vec3<i32>(1, 0, 1),
    vec3<i32>(1, 1, 0),
    vec3<i32>(1, 1, 1)
);

fn cell_contains_surface(cell_min: vec3<u32>) -> bool {
    var cell_min: vec3<i32> = vec3<i32>(cell_min);

    var num_negative: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var corner: vec3<i32> = cell_min + CORNERS[i];
        num_negative = num_negative + select(1u, 0u, textureLoad(sdf, corner).r < 0.0);
    }
    return num_negative != 0u && num_negative != 8u;
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var contains_surface: bool = cell_contains_surface(global_id);

    var surface_value: u32 = select(1u, 0u, contains_surface);
    textureStore(surface, vec3<i32>(global_id), vec4<u32>(surface_value, 0u, 0u, 1u));
}
