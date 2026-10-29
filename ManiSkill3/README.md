# ManiSkill 3 (Beta)


<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>


ManiSkill is a powerful unified framework for robot simulation and training powered by SAPIEN, with a strong focus on manipulation skills. The entire tech stack is as open-source as possible and ManiSkill v3 is in beta release now. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 30,000+ FPS with a 4090 GPU!
- GPU parallelized simulation, enabling high throughput state-based synthetic data collection in simulation
- GPU parallelized heterogeneous simulation, where every parallel environment has a completely different scene/set of objects
- Example tasks cover a wide range of different robot embodiments (humanoids, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, drawing/cleaning, dextrous manipulation)
- Flexible and simple task building API that abstracts away much of the complex GPU memory management code via an object oriented design
- Real2sim environments for scalably evaluating real-world policies 100x faster via GPU simulation.
- Many tuned robot learning baselines in Reinforcement Learning (e.g. PPO, SAC, TD-MPC2), Imitation Learning (e.g. Behavior Cloning, Diffusion Policy), and large Vision Language Action (VLA) models (e.g. Octo, RDT-1B, RT-x)

For more details we encourage you to take a look at our paper.

There are more features to be added to ManiSkill 3, see our roadmap for planned features that will be added over time before the official v3 is released.

Please refer to our documentation to learn more information from tutorials on building tasks to data collection.

**NOTE:**
This project currently is in a **beta release**, so not all features have been added in yet and there may be some bugs. If you find any bugs or have any feature requests please post them to our GitHub issues or discuss about them on GitHub discussions. We also have a Discord Server through which we make announcements and discuss about ManiSkill.

Users looking for the original ManiSkill2 can find the commit for that codebase at the v0.5.3 tag


## Installation
Installation of ManiSkill is extremely simple, you only need to run a few pip installs and setup Vulkan for rendering.

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```

Finally you also need to set up Vulkan with instructions here

Use the local installation instructions in the repository README.

## Getting Started

To get started, check out the quick start documentation:

We also have a quick start colab notebook that lets you try out GPU parallelized simulation without needing your own hardware. Everything is runnable on Colab free tier.

For a full list of example scripts you can run, see the docs.

## System Support

We currently best support Linux based systems. There is limited support for windows and no support for MacOS at the moment. We are working on trying to support more features on other systems but this may take some time. Most constraints stem from what the SAPIEN package is capable of supporting.

| System / GPU         | CPU Sim | GPU Sim | Rendering |
| -------------------- | ------- | ------- | --------- |
| Linux / NVIDIA GPU   | ✅      | ✅      | ✅        |
| Windows / NVIDIA GPU | ✅      | ❌      | ✅        |
| Windows / AMD GPU    | ✅      | ❌      | ✅        |
| WSL / Anything       | ✅      | ❌      | ❌        |
| MacOS / Anything     | ❌      | ❌      | ❌        |

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under CC BY-NC 4.0.
