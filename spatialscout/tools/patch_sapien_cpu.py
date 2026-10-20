"""Allow this pinned SAPIEN build to create a Mesa CPU Vulkan renderer.

SAPIEN 3.0.0b1 unconditionally requests VK_KHR_external_semaphore_fd,
which lavapipe does not expose. CPU rasterization never exports a CUDA
semaphore. The replacement repeats an available extension and preserves the
shared library's byte layout. Only use this patch for CPU rendering.
"""
from pathlib import Path
import sys
from spatialscout.paths import ROOT


def main():
    root = ROOT
    library = next((root / ".venv" / "lib").glob("python*/site-packages/sapien.libs/libsvulkan2.so"))
    old = b"VK_KHR_external_semaphore_fd\0"
    new = b"VK_KHR_external_memory_fd\0".ljust(len(old), b"\0")
    data = library.read_bytes()
    if data.count(old) == 1:
        backup = library.with_name("libsvulkan2.so.unpatched")
        if not backup.exists():
            backup.write_bytes(data)
        library.write_bytes(data.replace(old, new, 1))
        print(f"Patched {library}")
    elif new in data:
        print(f"Already patched: {library}")
    else:
        sys.exit("Unknown libsvulkan2 build; no patch applied")


if __name__ == "__main__":
    main()
