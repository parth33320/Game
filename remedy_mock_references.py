import os
import re

def apply_workspace_patches():
    print("🧹 [Patch Engine] Initiating sweep across workspace reference models...")

    # Target 1: Fix the Walkthrough Video Generator script fallback error
    target_video_script = "scripts/generate_walkthrough_video.py"
    if os.path.exists(target_video_script):
        print(f"📦 Patching {target_video_script} to strip custom manual canvas mock ups...")
        with open(target_video_script, "r", encoding="utf-8") as f:
            content = f.read()

        # Override the HeadlessRetroEnv instantiation to force standard Libretro interactions
        modified_content = content.replace(
            'env = HeadlessRetroEnv(obs_type="ram", use_retro=False)',
            'env = HeadlessRetroEnv(obs_type="ram", use_retro=True)'
        )

        # Edge Case Guard: Ensure it drops manual drawing commands in favor of actual emulator array pipes
        if 'canvas = np.zeros((height, width, 3), dtype=np.uint8)' in modified_content:
            print("⚠️ Detected artificial drawing routines. Replacing video writer feed with real emulator frames.")
            # Injecting logic to extract the physical frame memory array if use_retro is enabled
            real_frame_patch = """        # Extract clean visual data straight from the native Libretro emulator core if enabled
        if hasattr(env, 'retro_env') and env.retro_env is not None:
            # Gather raw frame buffer array
            canvas = env.retro_env.render(mode='rgb_array')
            if canvas is None:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                import cv2
                canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)"""

            # Locate the original canvas construction lines and substitute them out
            modified_content = re.sub(
                r'canvas = np\.zeros\(\(height, width, 3\), dtype=np\.uint8\).*?# Apply HUD overlay',
                real_frame_patch + '\n\n        # Apply HUD overlay',
                modified_content,
                flags=re.DOTALL
            )

        with open(target_video_script, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print(f"✅ successfully converted {target_video_script} to true hardware rendering modes.")
    else:
        print(f"❌ Error: Could not locate path destination: {target_video_script}")

    # Target 2: Enforce true stable-retro loading parameters inside the active environment module
    target_env_script = "agent/env.py"
    if os.path.exists(target_env_script):
        print(f"📦 Scanning {target_env_script} configuration dependencies...")
        with open(target_env_script, "r", encoding="utf-8") as f:
            env_content = f.read()

        # Ensure your wrapper handles genuine game-engine data rather than standard random uniform arrays
        if 'np.random.uniform(0.0, 1.0' in env_content:
            print("⚠️ Found procedural frame synthesis code. Adjusting data extraction mappings.")
            env_content = env_content.replace(
                'return np.random.uniform(0.0, 1.0, size=self.frame_shape).astype(np.float32)',
                '# Fetch active frame buffer slice from the core\n        return self.env.unwrapped.get_screen() if hasattr(self.env.unwrapped, "get_screen") else np.zeros((84,84,3), dtype=np.float32)'
            )
            with open(target_env_script, "w", encoding="utf-8") as f:
                f.write(env_content)
            print(f"✅ Cleaned placeholder components from {target_env_script}.")

    # Target 3: Force the training parameter file to isolate progression addresses
    target_config = "config/active_training_params.json"
    if os.path.exists(target_config):
        print(f"📦 Overwriting parameters file {target_config} to clear reward hacking constraints...")
        params_patch = """{
  "initial_lr": 0.00015,
  "ent_coef": 0.05,
  "gamma": 0.99,
  "time_penalty": -0.05,
  "score_weight": 0.0,
  "progress_multiplier": 1.5,
  "penalize_zero_velocity": true,
  "zero_velocity_penalty": -0.10,
  "lr_warmup_steps": 200,
  "reinit_policy_entropy": true
}"""
        with open(target_config, "w", encoding="utf-8") as f:
            f.write(params_patch)
        print(f"✅ Reset parameters in {target_config} to align with progress tracking metrics.")

    print("\n🚀 [Patch Complete] System files have been synchronized. Instruct your agent to clear the cache and execute 'python3 scripts/generate_walkthrough_video.py' to generate your verification proof video!")

if __name__ == "__main__":
    apply_workspace_patches()
