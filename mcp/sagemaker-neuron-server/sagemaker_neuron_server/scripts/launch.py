
"""Launcher: install latest optimum-neuron, then two-phase Neuron training."""
import os
import sys
import subprocess
import sysconfig

def main():
    nproc = os.environ.get("SM_NUM_NEURONS", "2")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    base_args = sys.argv[1:]

    # Install latest optimum-neuron and trl from main (DLC versions are behind the docs)
    print("=== Installing latest optimum-neuron and trl from main ===", flush=True)
    subprocess.run([  # nosemgrep: dangerous-subprocess-use-audit
        sys.executable, "-m", "pip", "install", "--upgrade",
        "git+https://github.com/huggingface/trl.git",
        "git+https://github.com/huggingface/optimum-neuron.git",
    ], check=True)  # nosemgrep: dangerous-subprocess-use-audit

    # Fix 1: optimum-neuron imports clone_chat_template from trl.models,
    # but in trl main it lives in trl.chat_template_utils.
    print("=== Patching optimum-neuron for trl compatibility ===", flush=True)
    site_dir = sysconfig.get_path("purelib")
    sft_path = os.path.join(site_dir, "optimum", "neuron", "trainers", "sft_trainer.py")
    with open(sft_path, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace("from trl.models import clone_chat_template", "from trl.chat_template_utils import clone_chat_template")
    with open(sft_path, "w", encoding="utf-8") as f:
        f.write(src)
    # Fix 2: optimum-neuron hard-pins trl==0.24.0 — remove the version check
    import_utils_path = os.path.join(site_dir, "optimum", "neuron", "utils", "import_utils.py")
    with open(import_utils_path, "r", encoding="utf-8") as f:
        src2 = f.read()
    src2 = src2.replace(
        'raise RuntimeError(f"Only `trl=={required_version}` is supported, but {trl_version} is installed.")',
        'pass  # version check disabled for latest trl'
    )
    with open(import_utils_path, "w", encoding="utf-8") as f:
        f.write(src2)
    print("Patches applied: clone_chat_template import + trl version check disabled", flush=True)

    # Phase 1: compile graphs with single process
    print(f"=== Phase 1: Compiling graphs with 1 process ===", flush=True)
    compile_env = os.environ.copy()
    compile_env["NEURON_EXTRACT_GRAPHS_ONLY"] = "1"
    compile_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", "1", "--nnodes", "1",
        "--master_addr", "localhost", "--master_port", "29500",
        train_script,
    ] + base_args
    print(f"Compile cmd: {' '.join(compile_cmd)}", flush=True)
    result = subprocess.run(compile_cmd, env=compile_env)  # nosemgrep: dangerous-subprocess-use-audit
    if result.returncode != 0:
        print(f"Compilation failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)

    # Phase 2: train with all NeuronCores
    print(f"=== Phase 2: Training with {nproc} processes ===", flush=True)
    train_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", nproc, "--nnodes", "1",
        "--master_addr", "localhost", "--master_port", "29500",
        train_script,
    ] + base_args
    print(f"Train cmd: {' '.join(train_cmd)}", flush=True)
    result = subprocess.run(train_cmd)  # nosemgrep: dangerous-subprocess-use-audit, dangerous-subprocess-use-tainted-env-args
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
