"""Apply the internal RHS-history correction before lossy storage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
cpu = ROOT / "src/DeepGPR/lib/deepgpr_cpu.c"
cuda = ROOT / "src/DeepGPR/lib/deepgpr.cu"

s = cpu.read_text()
start = s.index("/* Match update_e_cpu for one material cell/component.")
end = s.index("static void accumulate_material_gradients_cpu(", start)
helper = s[start:end]
s = s[:start] + s[end:]
pos = s.index("/*\n * Save the effective electric-field right-hand side R from")
s = s[:pos] + helper + s[pos:]
start = s.index("static void save_rhs_snapshot_cpu(")
end = s.index("/*\n * Copy one electric-field component snapshot", start)
body = s[start:end]
body = body.replace("HistoryBox box, int storage_kind)", "HistoryBox box, int storage_kind, int component)")
body = body.replace("        float rhs = cb_value != 0.0f\n            ? (E[field_idx] - ca[i * NY * NZ + j * NZ + k] * e_old) / cb_value",
                    "        float effective_ca = material_component_has_e_update_cpu(component, i, j, k, NX, NY, NZ)\n            ? ca[i * NY * NZ + j * NZ + k] : 1.0f;\n        float rhs = cb_value != 0.0f\n            ? (E[field_idx] - effective_ca * e_old) / cb_value")
s = s[:start] + body + s[end:]
old = """                } else if (!has_e_update && cb_value != 0.0f) {
                    /* Saved R was formed with ca even where E^n was unchanged.
                     * Remove that artificial history term; retain cb-dependent
                     * CPML corrections and source injection. */
                    rhs -= (e_old - ca_value * e_old) / cb_value;
"""
assert s.count(old) == 1
s = s.replace(old, "")
cursor = s.index("if (save_wavefield_history && store_rhs && i % sampling_interval == 0)")
for component in (0, 1, 2, 2):
    pos = s.index("history, storage_kind);", cursor)
    s = s[:pos] + f"history, storage_kind, {component});" + s[pos+len("history, storage_kind);"):]
    cursor = pos + 1
s = s.replace("* E^(n+1) = ca E^n + cb R. Eold_ptr contains E^n for this saved step.",
              "* E^(n+1) = a E^n + cb R, where a=ca for updated fields and a=1 otherwise.")
cpu.write_text(s)

s = cuda.read_text()
start = s.index("/* Match update_e_gpu for one material cell/component.")
end = s.index("template<int STORAGE_TYPE, int CONVERSION_BACKEND, int RHS_FROM_E>", start)
helper = s[start:end]
s = s[:start] + s[end:]
helper += """__device__ __forceinline__ float rhs_history_coefficient_gpu(
    int component, long long material_idx, const float* ca, int NX, int NY, int NZ)
{
    long long i = material_idx / ((long long)NY * NZ);
    long long remainder = material_idx % ((long long)NY * NZ);
    long long j = remainder / NZ, k = remainder % NZ;
    return material_component_has_e_update_gpu(component, i, j, k, NX, NY, NZ)
        ? ca[material_idx] : 1.0f;
}


"""
pos = s.index("/* Save R from the executed update E^(n+1) = ca E^n + cb R. */")
s = s[:pos] + helper + s[pos:]
for name, end_marker in (("save_rhs_snapshot_gpu", "__device__ __forceinline__ void compact_snapshot_indices"),
                         ("save_rhs_snapshot_vec2_gpu", "/*\n * Quantize one spatial tile")):
    start = s.index(f"__global__ void {name}(")
    end = s.index(end_marker, start)
    body = s[start:end]
    assert body.count("HistoryBox box)") == 1
    body = body.replace("HistoryBox box)", "HistoryBox box, int component)")
    for material in ("material_idx", "material0", "material1"):
        body = body.replace(f"ca[{material}] *", f"rhs_history_coefficient_gpu(component, {material}, ca, NX, NY, NZ) *")
    s = s[:start] + body + s[end:]
start = s.index("__global__ void quantize_rhs_int8_snapshot_gpu(")
end = s.index("/*", start)
body = s[start:end]
assert body.count("ca[material_idx] *") == 1
body = body.replace("ca[material_idx] *", "rhs_history_coefficient_gpu(components == 3 ? component : 2, material_idx, ca, NX, NY, NZ) *")
s = s[:start] + body + s[end:]
old = """                    if (!has_e_update && cb_value != 0.0f) {
                        rhs -= (e_old - ca_value * e_old) / cb_value;
                    }
"""
assert s.count(old) == 1
s = s.replace(old, "")
old = """                if (!has_e_update && cb_value != 0.0f) {
                    rhs -= (e_old - ca_value * e_old) / cb_value;
                }
"""
assert s.count(old) == 1
s = s.replace(old, "")
cursor = s.index("} else if (use_async) {", s.index("quantize_rhs_int8_snapshot_gpu,", s.index("void deepgpr_forward")) if "void deepgpr_forward" in s else s.index("quantize_rhs_int8_snapshot_gpu,"))
for component in (0, 1, 2, 2, 0, 1, 2, 2):
    cursor = s.index("LAUNCH_SAVE_KERNEL(save_rhs_snapshot_gpu", cursor)
    pos = s.index("history);", cursor)
    s = s[:pos] + f"history, {component});" + s[pos+len("history);"):]
    cursor = pos + 1
cuda.write_text(s)
