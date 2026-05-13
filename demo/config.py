from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    max_queue_size: int
    timeout: float
    ssl_certfile: str
    ssl_keyfile: str
    config_path: str
    checkpoint_folder: str
    step: int
    noise_scale: float
    debug: bool
    num_gpus: int
    gpu_ids: str
    max_outstanding: int
    schedule_block: bool
    model_type: str
    use_taehv: bool
    use_tensorrt: bool
    fast: bool
    conditioning_source: str
    semantic_avatar_target_fps: int
    semantic_avatar_jpeg_quality: int
    semantic_avatar_max_input_queue_frames: int
    semantic_avatar_image_format: str
    debug_semantic_overlay: bool
    debug_face_mask: bool
    enable_metrics: bool
    target_latency: float
    t2v: bool

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_ROOT)

default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument(
    "--ssl-certfile",
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")

# This is the default config for the pipeline, it can be overridden by the command line arguments
parser.add_argument(
    "--config_path",
    type=str,
    default=os.path.join(PROJECT_ROOT, "configs", "wan_causal_dmd_v2v.yaml"),
)
parser.add_argument(
    "--checkpoint_folder",
    type=str,
    default=os.path.join(PROJECT_ROOT, "ckpts", "wan_causal_dmd_v2v"),
)
parser.add_argument("--step", type=int, default=2)
parser.add_argument("--noise_scale", type=float, default=0.8)
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--gpu_ids", type=str, default="0,1") # id separated by comma, size should match num_gpus

# These are only used when num_gpus > 1
parser.add_argument("--max_outstanding", type=int, default=2, help="max number of outstanding sends/recv to keep")
parser.add_argument("--schedule_block", action="store_true", default=False)
parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
parser.add_argument("--use_taehv", action="store_true", default=os.getenv("USE_TAEHV", "").lower() in {"1", "true", "yes", "on"}, help="Use the TAEHV decoder for online inference")
parser.add_argument("--use_tensorrt", action="store_true", default=os.getenv("USE_TENSORRT", "").lower() in {"1", "true", "yes", "on"}, help="Enable available TensorRT acceleration paths for online inference")
parser.add_argument("--fast", action="store_true", default=os.getenv("FAST", "").lower() in {"1", "true", "yes", "on"}, help="Enable the fast path: --use_taehv --use_tensorrt")
parser.add_argument(
    "--conditioning_source",
    choices=("rgb", "semantic_pose"),
    default=os.getenv("CONDITIONING_SOURCE", "rgb"),
    help="Input conditioning source. 'rgb' preserves upstream video behavior; 'semantic_pose' renders server-side pose maps from semantic packets.",
)
parser.add_argument(
    "--semantic-avatar-target-fps",
    dest="semantic_avatar_target_fps",
    type=int,
    default=int(os.getenv("SEMANTIC_AVATAR_TARGET_FPS", "8")),
    help="Target websocket output cadence for /ws/semantic-avatar.",
)
parser.add_argument(
    "--semantic-avatar-jpeg-quality",
    dest="semantic_avatar_jpeg_quality",
    type=int,
    default=int(os.getenv("SEMANTIC_AVATAR_JPEG_QUALITY", "68")),
    help="JPEG/WebP quality for semantic avatar websocket frames.",
)
parser.add_argument(
    "--semantic-avatar-max-input-queue-frames",
    dest="semantic_avatar_max_input_queue_frames",
    type=int,
    default=int(os.getenv("SEMANTIC_AVATAR_MAX_INPUT_QUEUE_FRAMES", "16")),
    help="Maximum queued synthetic driving frames before stale frames are dropped.",
)
parser.add_argument(
    "--semantic-avatar-image-format",
    dest="semantic_avatar_image_format",
    choices=("JPEG", "WEBP"),
    default=os.getenv("SEMANTIC_AVATAR_IMAGE_FORMAT", "JPEG").upper(),
    help="Binary websocket frame image format.",
)
parser.add_argument(
    "--debug-semantic-overlay",
    dest="debug_semantic_overlay",
    action="store_true",
    default=os.getenv("DEBUG_SEMANTIC_OVERLAY", "").lower() in {"1", "true", "yes", "on"},
    help="Render semantic maps visibly in avatar conditioning frames for debugging. Off by default.",
)
parser.add_argument(
    "--debug-face-mask",
    dest="debug_face_mask",
    action="store_true",
    default=os.getenv("DEBUG_FACE_MASK", "").lower() in {"1", "true", "yes", "on"},
    help="Render localized face/body masks and semantic debug labels. Off by default.",
)

# Metrics collection
parser.add_argument("--enable-metrics", dest="enable_metrics", action="store_true", default=False, help="Enable SLO metrics collection")
parser.add_argument("--target-latency", dest="target_latency", type=float, default=1.0, help="Target latency in seconds for deadline miss rate calculation (default: 0.5s)")
parser.add_argument("--t2v", action="store_true", default=False)

parsed_args = vars(parser.parse_args())
parsed_args["config_path"] = os.path.abspath(parsed_args["config_path"])
parsed_args["checkpoint_folder"] = os.path.abspath(parsed_args["checkpoint_folder"])

gpu_ids = [gpu_id.strip() for gpu_id in parsed_args["gpu_ids"].split(",") if gpu_id.strip()]
if len(gpu_ids) != parsed_args["num_gpus"]:
    raise ValueError(
        f"--gpu_ids expects {parsed_args['num_gpus']} entries, got {len(gpu_ids)} from '{parsed_args['gpu_ids']}'"
    )
parsed_args["gpu_ids"] = ",".join(gpu_ids)

config = Args(**parsed_args)
