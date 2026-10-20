"""Save ManiSkill camera observations without opening a window."""
import argparse
import os
from pathlib import Path

from spatialscout.paths import ROOT
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / "assets"))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/maniskill-runtime")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)
# The H100 MIG device on this host has no exposed Vulkan graphics device.
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
os.environ.setdefault("LP_NUM_THREADS", "8")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import gymnasium as gym
import mani_skill.envs
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "scene.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="Random control steps before capture")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.steps < 0 or args.size < 1:
        parser.error("steps must be >= 0 and size must be >= 1")

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
        sensor_configs={"width": args.size, "height": args.size, "shader_pack": "minimal"},
        human_render_camera_configs={"width": args.size, "height": args.size, "shader_pack": "minimal"},
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        for _ in range(args.steps):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if bool(terminated.any()) or bool(truncated.any()):
                break
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rgb = env.render().detach().cpu().numpy()[0]
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(args.output)
        print(f"Scene camera: {args.output.resolve()}")
        for name, data in obs["sensor_data"].items():
            path = args.output.with_name(f"{args.output.stem}_{name}.png")
            Image.fromarray(data["rgb"][0].detach().cpu().numpy().astype(np.uint8)).save(path)
            print(f"Sensor {name}: {path.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
