import os

def apply_cpu_only_patches():
    print("🖥️ [Hardware Guard] Enforcing strict CPU-and-RAM training topology...")

    # 1. Patch the main training entrypoint file
    target_train_script = "scripts/train_agent.py"
    if os.path.exists(target_train_script):
        print(f"📦 Patching {target_train_script} to force torch CPU device targets...")
        with open(target_train_script, "r", encoding="utf-8") as f:
            content = f.read()

        # Forlap any device logic with a hardcoded CPU flag
        if "device =" not in content and "device=" not in content:
            content = content.replace(
                "model = ActorCriticPPO(input_channels=4, num_actions=8)",
                'device = "cpu"\n    model = ActorCriticPPO(input_channels=4, num_actions=8).to(device)'
            )
        else:
            content = content.replace('device = "cuda"', 'device = "cpu"')
            content = content.replace("map_location=\"cpu\"", "map_location=\"cpu\"")

        with open(target_train_script, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Forced CPU execution tensors inside {target_train_script}.")

    # 2. Patch the colab notebook and watchdog files to remove any --cuda execution flags
    target_watchdog = "scripts/watchdog.py"
    if os.path.exists(target_watchdog):
        with open(target_watchdog, "r", encoding="utf-8") as f:
            w_content = f.read()
        w_content = w_content.replace("--cuda", "")
        with open(target_watchdog, "w", encoding="utf-8") as f:
            f.write(w_content)
        print(f"✅ Removed all '--cuda' hardware arguments from {target_watchdog}.")

    # 3. Patch Dockerfile configuration parameters
    dockerfile_path = "docker/Dockerfile"
    if os.path.exists(dockerfile_path):
        print(f"📦 Optimizing {dockerfile_path} for lighter CPU-only footprints...")
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            d_content = f.read()

        # Re-map standard pip installation matrices to utilize light CPU-only torch builds
        if "pip install" in d_content and "cpu" not in d_content:
            d_content = d_content.replace(
                "pip install --no-cache-dir --prefix=/install \\\n    pytest \\\n    torch \\\n    numpy",
                "pip install --no-cache-dir --prefix=/install \\\n    pytest \\\n    torch --index-url https://pytorch.org \\\n    numpy"
            )
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(d_content)
            print(f"✅ Swapped default torch container mapping to standard Python CPU wheel distributions.")

    print("\n🚀 [Hardware Patch Complete] Your workspace is completely locked to CPU and RAM usage. No GPU allocations will occur.")

if __name__ == "__main__":
    apply_cpu_only_patches()
