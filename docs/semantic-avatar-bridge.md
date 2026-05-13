# Semantic Avatar Bridge

This bridge adds an incremental semantic-pose conditioning path without changing the upstream RGB path.

## Flag

```bash
python demo/main.py --conditioning_source rgb
python demo/main.py --conditioning_source semantic_pose
```

`rgb` is the default and preserves upstream behavior. `semantic_pose` expects JSON semantic packets in the existing websocket parameter message instead of image bytes.

## Semantic Packet Schema

Minimum useful packet:

```json
{
  "t": 1715000000,
  "frameId": 123,
  "yaw": 12.4,
  "pitch": -3.2,
  "roll": 1.1,
  "headX": 0.5,
  "headY": 0.35,
  "shoulderX": 0.5,
  "shoulderY": 0.62,
  "confidence": 0.94
}
```

The bridge also accepts snake_case names and optional `poseLandmarks` / `pose_landmarks` / `landmarks` as normalized `[x, y]` points or objects with `x` and `y`.

The demo websocket can send this either as:

```json
{ "semantic_packet": { "yaw": 12.4, "headX": 0.5, "headY": 0.35 } }
```

or as top-level semantic fields; the backend wraps known semantic fields into `semantic_packet`.

## Conditioning Image Format

- RGB image-like array.
- Shape before queue stacking: `[height, width, 3]`.
- Value range: normalized float array in `[-1.0, 1.0]`.
- Tensor shape after `demo/util.py::read_images_from_queue`: `[1, 3, T, H, W]`.
- Contents: black background with lightweight head, neck, shoulder, and head-axis strokes. Mouth controls are intentionally ignored.

## Queue Flow

```text
websocket params JSON
  -> Pipeline.InputParams.semantic_packet
  -> SemanticPoseConditioningAdapter.packet_to_array
  -> input_queue normalized pose-map frame
  -> read_images_from_queue
  -> SingleGPUInferencePipeline.start_stream_session / run_stream_batch
  -> _prepare_conditioning_images
  -> VAE stream_encode
```

## Debug Hooks

Set these on the backend process:

```bash
SEMANTIC_POSE_DEBUG_DIR=debug/pose_maps
SEMANTIC_POSE_DEBUG_EVERY_N=30
```

The adapter writes periodic PNG pose maps and logs render timing. Runtime metrics are also logged every 120 rendered semantic pose frames.

## FPS Impact

Pose-map rendering is CPU-side PIL line drawing and should usually be far below the VAE and DiT cost. Expect less than 1 ms per frame on typical RunPod CPUs at 512x512, with practical FPS still dominated by StreamDiffusionV2 encode, denoise, and decode.
