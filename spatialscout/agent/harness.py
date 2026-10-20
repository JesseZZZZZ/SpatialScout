"""Let a vision language model explore ManiSkill through a movable camera."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from spatialscout.paths import ROOT
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / "assets"))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/maniskill-runtime")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS", "8")

import gymnasium as gym
import mani_skill.envs  # Register the environments.
import numpy as np
import requests
from PIL import Image
from mani_skill.utils import sapien_utils


MAX_API_RETRIES = 10
MAX_API_RETRY_DELAY_SECONDS = 30
RETRYABLE_API_STATUSES = (429, 500, 502, 503, 504)


def wait_before_api_retry(attempt: int) -> None:
    time.sleep(min(2 ** attempt, MAX_API_RETRY_DELAY_SECONDS))


SYSTEM_PROMPT = """You are exploring a simulated robot work space through a free-moving camera.
You can move and rotate the camera, read its intrinsics and pose, and capture its current image.
All positions are world coordinates in meters; z is up. Yaw and pitch are degrees.
The camera can move through objects because it is a virtual observation camera.
Use capture_image to see the current view and call it again after changing the camera.
Use image evidence when describing the scene. Do not claim to have moved the robot or changed
the physical scene; these tools only move the camera. Finish with a concise answer to the user.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_camera_pose",
            "description": "Read the camera's current world position and viewing angles.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_intrinsics",
            "description": "Read the actual 3x3 camera intrinsic matrix, image size, and clipping planes.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera_position",
            "description": "Teleport the virtual camera to an absolute world position in meters, keeping its orientation.",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "number", "description": f"World {k} coordinate in meters"} for k in ("x", "y", "z")},
                "required": ["x", "y", "z"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_camera",
            "description": "Move relative to the camera: forward along the horizontal view direction, right horizontally, up along world z. Each move is limited to 1 meter.",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "number", "description": "Distance in meters"} for k in ("forward", "right", "up")},
                "required": ["forward", "right", "up"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_camera",
            "description": "Rotate the camera by relative yaw and pitch in degrees. Positive yaw turns counterclockwise viewed from above; positive pitch looks upward.",
            "parameters": {
                "type": "object",
                "properties": {"yaw_deg": {"type": "number"}, "pitch_deg": {"type": "number"}},
                "required": ["yaw_deg", "pitch_deg"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Capture the camera's current RGB image. The PNG is saved locally and the image is sent to your vision input.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class CameraWorld:
    def __init__(self, env_id: str, size: int, seed: int, output_dir: Path, env_kwargs: dict | None = None, position_bound: float = 3):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.position_bound = position_bound
        self.env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="none",
            render_mode="rgb_array",
            sim_backend="cpu",
            render_backend="cpu",
            human_render_camera_configs={"width": size, "height": size, "shader_pack": "minimal"},
            **(env_kwargs or {}),
        )
        self.env.reset(seed=seed)
        cameras = self.env.unwrapped._human_render_cameras
        if not cameras:
            self.env.close()
            raise RuntimeError(f"{env_id} has no render camera")
        self.camera_name = "render_camera" if "render_camera" in cameras else next(iter(cameras))
        self.camera = cameras[self.camera_name].camera
        raw_pose = self.camera._render_cameras[0].get_local_pose()
        self.position = np.array(raw_pose.p, dtype=float)
        forward = self.camera.get_model_matrix()[0, :3, 2].cpu().numpy()
        # The model matrix uses OpenGL axes; negative local z is the view direction.
        forward = -forward
        self.yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
        self.pitch = math.degrees(math.atan2(float(forward[2]), float(np.linalg.norm(forward[:2]))))
        self.frame_index = 0

    def close(self):
        self.env.close()

    def pose(self) -> dict:
        return {
            "camera": self.camera_name,
            "position_m": [round(float(x), 5) for x in self.position],
            "yaw_deg": round(self.yaw, 3),
            "pitch_deg": round(self.pitch, 3),
            "coordinate_frame": "world xyz, z up",
        }

    def _apply_pose(self):
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        direction = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        target = self.position + direction
        pose = sapien_utils.look_at(self.position, target)
        self.camera.set_local_pose(pose.sp)

    def set_position(self, x: float, y: float, z: float) -> dict:
        candidate = np.array([finite_number(x, "x"), finite_number(y, "y"), finite_number(z, "z")])
        if max(abs(candidate[0]), abs(candidate[1]), abs(candidate[2])) > self.position_bound:
            raise ValueError(f"Camera world coordinates must stay within ±{self.position_bound} m")
        self.position = candidate
        self._apply_pose()
        return self.pose()

    def move(self, forward: float, right: float, up: float) -> dict:
        distance = np.array([finite_number(forward, "forward"), finite_number(right, "right"), finite_number(up, "up")])
        if np.linalg.norm(distance) > 1:
            raise ValueError("Maximum movement per call is 1 meter")
        yaw = math.radians(self.yaw)
        forward_axis = np.array([math.cos(yaw), math.sin(yaw), 0])
        right_axis = np.array([math.sin(yaw), -math.cos(yaw), 0])
        delta = distance[0] * forward_axis + distance[1] * right_axis + distance[2] * np.array([0, 0, 1])
        return self.set_position(*(self.position + delta))

    def rotate(self, yaw_deg: float, pitch_deg: float) -> dict:
        yaw_delta = finite_number(yaw_deg, "yaw_deg")
        pitch_delta = finite_number(pitch_deg, "pitch_deg")
        if max(abs(yaw_delta), abs(pitch_delta)) > 180:
            raise ValueError("Maximum turn per call is 180 degrees per axis")
        self.yaw = (self.yaw + yaw_delta + 180) % 360 - 180
        self.pitch = max(-85, min(85, self.pitch + pitch_delta))
        self._apply_pose()
        return self.pose()

    def intrinsics(self) -> dict:
        k = self.camera.get_intrinsic_matrix()[0].cpu().numpy()
        return {
            "camera": self.camera_name,
            "width": self.camera.get_width(),
            "height": self.camera.get_height(),
            "K": k.tolist(),
            "fx": float(k[0, 0]), "fy": float(k[1, 1]),
            "cx": float(k[0, 2]), "cy": float(k[1, 2]),
            "near_m": self.camera.get_near(), "far_m": self.camera.get_far(),
        }

    def capture(self) -> tuple[dict, Path]:
        started = time.perf_counter()
        frame = self.env.unwrapped.render_rgb_array(camera_name=self.camera_name)
        if frame is None:
            raise RuntimeError("Camera returned no image")
        array = frame[0].detach().cpu().numpy().astype(np.uint8)
        path = self.output_dir / f"frame_{self.frame_index:04d}.png"
        Image.fromarray(array).save(path)
        self.frame_index += 1
        return {"path": str(path), "width": int(array.shape[1]), "height": int(array.shape[0]), "pose": self.pose(), "capture_seconds": round(time.perf_counter() - started, 3)}, path

    def call(self, name: str, args: dict) -> tuple[dict, Path | None]:
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a JSON object")
        names = {
            "get_camera_pose": ([], self.pose),
            "get_camera_intrinsics": ([], self.intrinsics),
            "set_camera_position": (["x", "y", "z"], self.set_position),
            "move_camera": (["forward", "right", "up"], self.move),
            "rotate_camera": (["yaw_deg", "pitch_deg"], self.rotate),
            "capture_image": ([], self.capture),
        }
        if name not in names:
            raise ValueError(f"Unknown tool: {name}")
        keys, fn = names[name]
        if set(args) != set(keys):
            raise ValueError(f"{name} requires exactly: {', '.join(keys) or 'no arguments'}")
        result = fn(**args)
        return result if name == "capture_image" else (result, None)


def api_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise ValueError("Base URL must begin with https:// or http://")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/responses"):
        base = base[: -len("/responses")]
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def chat_url(base_url: str) -> str:
    return f"{api_root(base_url)}/chat/completions"


def responses_url(base_url: str) -> str:
    return f"{api_root(base_url)}/responses"


def anthropic_messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        base = base[: -len("/messages")]
    return f"{api_root(base)}/messages"


def gemini_generate_content_url(base_url: str, model: str) -> str:
    base = base_url.rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise ValueError("Base URL must begin with https:// or http://")
    if base.endswith(":generateContent"):
        return base
    if base.endswith("/v1"):
        base = base[: -len("/v1")] + "/v1beta"
    elif not base.endswith(("/v1beta", "/v1beta/models")):
        base += "/v1beta"
    if not base.endswith("/models"):
        base += "/models"
    model_id = model.removeprefix("models/")
    return f"{base}/{quote(model_id, safe='-._~')}:generateContent"


def resolve_model(base_url: str, key: str, requested: str | None, timeout: int) -> str:
    if requested:
        return requested
    base = api_root(base_url)
    try:
        response = requests.get(
            f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout
        )
        response.raise_for_status()
        ids = [entry["id"] for entry in response.json()["data"] if isinstance(entry.get("id"), str)]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Could not discover a model from /models; specify --model or VLM_MODEL") from exc
    if len(ids) == 1:
        return ids[0]
    vision = [name for name in ids if any(term in name.lower() for term in ("vision", "vlm", "-vl", "gpt-4o", "gpt-4.1", "gpt-5"))]
    if len(vision) == 1:
        return vision[0]
    raise RuntimeError(f"The API lists {len(ids)} models; specify a vision model with --model or VLM_MODEL")


def completion(url: str, key: str, model: str, messages: list, tools: list, timeout: int, include_usage: bool = False) -> dict | tuple[dict, dict | None]:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload.update({"tools": tools, "tool_choice": "auto"})
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            return (message, data.get("usage")) if include_usage else message
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in RETRYABLE_API_STATUSES and attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            body = exc.response.text[:600] if exc.response is not None else ""
            retries = f" after {MAX_API_RETRIES} retries" if status in RETRYABLE_API_STATUSES else ""
            raise RuntimeError(f"VLM API returned HTTP {status}{retries}: {body}") from exc
        except requests.RequestException as exc:
            if attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            raise RuntimeError(f"VLM API request failed after {MAX_API_RETRIES} retries: {exc}") from exc
        except (ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(f"VLM API response could not be decoded: {exc}") from exc


def responses_completion(url: str, key: str, model: str, input_items: list, tools: list, timeout: int, previous_response_id: str | None = None, reasoning_effort: str = "high") -> dict:
    response_tools = [{"type": "function", **tool["function"]} for tool in tools]
    payload = {
        "model": model, "input": input_items, "tools": response_tools,
        "tool_choice": "auto", "reasoning": {"effort": reasoning_effort},
        "store": True,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data.get("output"), list) or not data.get("id"):
                raise ValueError("Response has no id or output")
            return data
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in RETRYABLE_API_STATUSES and attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            body = exc.response.text[:600] if exc.response is not None else ""
            retries = f" after {MAX_API_RETRIES} retries" if status in RETRYABLE_API_STATUSES else ""
            raise RuntimeError(f"VLM API returned HTTP {status}{retries}: {body}") from exc
        except requests.RequestException as exc:
            if attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            raise RuntimeError(f"VLM API request failed after {MAX_API_RETRIES} retries: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"VLM API response could not be decoded: {exc}") from exc


def anthropic_completion(url: str, key: str, model: str, messages: list, tools: list, timeout: int, system: str, reasoning_effort: str = "high") -> dict:
    anthropic_tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"],
        }
        for tool in tools
    ]
    payload = {
        "model": model,
        "max_tokens": 16384,
        "system": system,
        "messages": messages,
        "tools": anthropic_tools,
        "tool_choice": {"type": "auto"},
    }
    if reasoning_effort != "none":
        payload["output_config"] = {"effort": reasoning_effort}
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data.get("content"), list) or not data.get("id"):
                raise ValueError("Message has no id or content")
            return data
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in RETRYABLE_API_STATUSES and attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            body = exc.response.text[:600] if exc.response is not None else ""
            retries = f" after {MAX_API_RETRIES} retries" if status in RETRYABLE_API_STATUSES else ""
            raise RuntimeError(f"VLM API returned HTTP {status}{retries}: {body}") from exc
        except requests.RequestException as exc:
            if attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            raise RuntimeError(f"VLM API request failed after {MAX_API_RETRIES} retries: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"VLM API response could not be decoded: {exc}") from exc


def gemini_completion(url: str, key: str, contents: list, tools: list, timeout: int, system: str, reasoning_effort: str) -> dict:
    declarations = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parametersJsonSchema": tool["function"]["parameters"],
        }
        for tool in tools
    ]
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "tools": [{"functionDeclarations": declarations}],
        "toolConfig": {"functionCallingConfig": {"mode": "VALIDATED"}},
        "generationConfig": {
            "candidateCount": 1,
            "maxOutputTokens": 32768,
            "thinkingConfig": {"thinkingLevel": reasoning_effort.upper()},
        },
    }
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates")
            content = candidates[0].get("content") if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else None
            if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
                raise ValueError(f"Response has no candidate content parts; promptFeedback={data.get('promptFeedback')}; finishReason={candidates[0].get('finishReason') if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else None}")
            return data
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in RETRYABLE_API_STATUSES and attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            body = exc.response.text[:600] if exc.response is not None else ""
            retries = f" after {MAX_API_RETRIES} retries" if status in RETRYABLE_API_STATUSES else ""
            raise RuntimeError(f"VLM API returned HTTP {status}{retries}: {body}") from exc
        except requests.RequestException as exc:
            if attempt < MAX_API_RETRIES:
                wait_before_api_retry(attempt)
                continue
            raise RuntimeError(f"VLM API request failed after {MAX_API_RETRIES} retries: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"VLM API response could not be decoded: {exc}") from exc


def image_message(path: Path) -> dict:
    # Keep the original PNG on disk; send a compact copy to the VLM service.
    image = Image.open(path).convert("RGB")
    image.thumbnail((384, 384))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Camera image {path.name}, recorded when capture_image was called. Later camera moves may have changed the current view. Continue exploring or answer the task."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ],
    }


def responses_image_message(path: Path) -> dict:
    chat_image = image_message(path)
    parts = chat_image["content"]
    return {"role": "user", "content": [
        {"type": "input_text", "text": parts[0]["text"]},
        {"type": "input_image", "image_url": parts[1]["image_url"]["url"], "detail": "auto"},
    ]}


def gemini_image_message(path: Path) -> dict:
    chat_image = image_message(path)
    parts = chat_image["content"]
    encoded = parts[1]["image_url"]["url"].split(",", 1)[1]
    return {"role": "user", "parts": [
        {"text": parts[0]["text"]},
        {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
    ]}


def anthropic_image_message(path: Path) -> dict:
    chat_image = image_message(path)
    parts = chat_image["content"]
    encoded = parts[1]["image_url"]["url"].split(",", 1)[1]
    return {"role": "user", "content": [
        {"type": "text", "text": parts[0]["text"]},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}},
    ]}


def run_agent(world: CameraWorld, url: str, key: str, model: str, task: str, max_turns: int, timeout: int):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    log_path = world.output_dir / "agent_trace.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for turn in range(max_turns):
            message = completion(url, key, model, messages, TOOLS, timeout)
            calls = message.get("tool_calls") or []
            content = message.get("content")
            log.write(json.dumps({"turn": turn, "assistant": {"content": content, "tool_calls": calls}}, ensure_ascii=False) + "\n")
            log.flush()
            if not calls:
                print(content or "(The model returned an empty response.)")
                print(f"Images and trace: {world.output_dir}")
                return
            if not isinstance(calls, list):
                raise RuntimeError("VLM returned malformed tool_calls")
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": call["function"]}
                    for call in calls
                ],
            })
            new_images = []
            for call in calls:
                name = call.get("function", {}).get("name", "")
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    result, image_path = world.call(name, args)
                    result_text = json.dumps(result, ensure_ascii=False)
                    if image_path:
                        new_images.append(image_path)
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    result_text = json.dumps({"error": str(exc)}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result_text})
                log.write(json.dumps({"tool": name, "result": json.loads(result_text)}, ensure_ascii=False) + "\n")
                print(f"{name}: {result_text}", file=sys.stderr)
            for path in new_images:
                messages.append(image_message(path))
            log.flush()

        messages.append({"role": "user", "content": "Tool budget reached. Summarize what you observed now."})
        final = completion(url, key, model, messages, [], timeout)
        print(final.get("content") or "(The model returned an empty response.)")
        print(f"Images and trace: {world.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL"), help="OpenAI-compatible API base, usually ending in /v1")
    parser.add_argument("--api-key", default=os.getenv("VLM_API_KEY"), help="API key; VLM_API_KEY is safer than a CLI argument")
    parser.add_argument("--model", default=os.getenv("VLM_MODEL"), help="Vision model ID served by the API")
    parser.add_argument("--task", default="Explore the robot work space from several camera views and describe the robot, table, and visible objects.")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "agent_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--camera-demo", action="store_true", help="Capture one frame and print camera parameters without contacting a VLM")
    args = parser.parse_args()
    if args.size < 64 or args.size > 1024 or args.max_turns < 1 or args.timeout < 1:
        parser.error("size must be 64..1024, max-turns and timeout must be positive")
    if not args.camera_demo and not all((args.base_url, args.api_key)):
        parser.error("provide --base-url and --api-key, or set VLM_BASE_URL and VLM_API_KEY")
    try:
        url = chat_url(args.base_url) if not args.camera_demo else ""
        model = resolve_model(args.base_url, args.api_key, args.model, args.timeout) if not args.camera_demo else ""
        world = CameraWorld(args.env_id, args.size, args.seed, args.output_dir)
        try:
            if args.camera_demo:
                info, _ = world.capture()
                print(json.dumps({"pose": world.pose(), "intrinsics": world.intrinsics(), "image": info}, indent=2))
            else:
                run_agent(world, url, args.api_key, model, args.task, args.max_turns, args.timeout)
        finally:
            world.close()
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
