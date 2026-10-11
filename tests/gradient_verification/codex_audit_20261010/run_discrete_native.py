"""Run existing focused discrete-adjoint tests on the isolated fixed library."""
import argparse
import io
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import test_discrete_adjoint as existing
from DeepGPR.native import loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    loader.LIB_DIR = HERE / "native_libraries/after"
    loader._LOADED_LIBS.clear()
    names = [
        "test_weighted_curl_dot_products_orders_2_4_8",
        "test_native_field_update_dot_products_orders_2_4_8",
        "test_native_cpml_single_step_dot_product_all_faces",
        "test_source_waveform_gradient_matches_finite_difference",
        "test_material_taylor_checks_without_and_with_cpml",
        "test_incomplete_sampling_block_uses_actual_length",
        "test_native_2d_tm_state_dot_products_orders_2_4_8",
    ] if args.device == "cpu" else [
        "test_cuda_cpml_single_step_dot_products_orders_2_4_8",
        "test_cuda_asymmetric_cpml_dot_products_orders_2_4_8",
        "test_cuda_2d_tm_adjoint_is_bitwise_reproducible",
        "test_cpu_cuda_forward_and_gradient_consistency",
        "test_cuda_coincident_sources_and_receivers_accumulate",
        "test_cuda_2d_tm_fast_path_state_dot_products_orders_2_4_8",
        "test_cuda_material_taylor_checks_without_and_with_cpml",
        "test_cuda_2d_tm_fast_path_selection_and_forward_identity",
        "test_cuda_asymmetric_cpml_and_incomplete_sampling_block",
    ]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.TestSuite(existing.DiscreteAdjointTests(name) for name in names))
    log = stream.getvalue()
    (HERE / f"discrete_native_{args.device}.log").write_text(log)
    payload = {"device": args.device, "tests": names, "ran": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    (HERE / f"discrete_native_{args.device}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(log)
    print(json.dumps(payload, indent=2))
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
