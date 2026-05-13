# Semantic Avatar Realtime Integration

This integration keeps StreamDiffusionV2 upstream paths intact. The browser sends
MediaPipe semantic packets, the new `semantic_avatar` adapter converts them into
synthetic portrait-driving frames, and the existing StreamV2V worker consumes
those frames through its normal image-conditioning queue.

## Backend Routes

- `POST /avatar/upload`
  - Accepts JPG, PNG, or WebP.
  - Resizes/crops the portrait to the active model resolution, default `512x512`.
  - Caches an in-memory portrait session and returns `avatar_id`.

- `WS /ws/semantic-avatar?avatar_id=<id>`
  - Accepts JSON semantic packets.
  - Uses latest-packet-wins semantics.
  - Sends generated avatar frames as binary JPEG/WebP messages.
  - Sends JSON metrics once per second.

- `GET /semantic-avatar/health`
  - Returns route, adapter, and stream metrics.

## Startup

Local single-GPU target:

```shell
cd StreamDiffusionV2
python demo/main.py \
  --host 0.0.0.0 \
  --port 7860 \
  --num_gpus 1 \
  --gpu_ids 0 \
  --step 1 \
  --noise_scale 0.65 \
  --semantic-avatar-target-fps 8 \
  --semantic-avatar-jpeg-quality 68
```

Lower latency RunPod target:

```shell
cd StreamDiffusionV2
USE_TAEHV=1 SEMANTIC_AVATAR_TARGET_FPS=6 python demo/main.py \
  --host 0.0.0.0 \
  --port 7860 \
  --num_gpus 1 \
  --gpu_ids 0 \
  --step 1 \
  --noise_scale 0.55 \
  --semantic-avatar-max-input-queue-frames 12
```

The normal `/api/ws/{user_id}` and `/api/stream/{user_id}` demo routes still
exist. For the semantic avatar milestone, use only one active semantic-avatar
client per backend process because it shares the current StreamV2V worker queue.

## Minimal Frontend Example

```js
async function connectSemanticAvatar(baseUrl, portraitFile, getSemanticPacket) {
  const form = new FormData();
  form.append("image", portraitFile);

  const upload = await fetch(new URL("/avatar/upload", baseUrl), {
    method: "POST",
    body: form
  });
  if (!upload.ok) throw new Error(await upload.text());

  const { avatar_id } = await upload.json();
  const wsUrl = new URL("/ws/semantic-avatar", baseUrl);
  wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
  wsUrl.searchParams.set("avatar_id", avatar_id);

  const ws = new WebSocket(wsUrl);
  ws.binaryType = "blob";

  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
      console.log("semantic-avatar metrics", JSON.parse(event.data));
      return;
    }

    const bitmap = await createImageBitmap(event.data);
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
  };

  const timer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(getSemanticPacket()));
  }, 1000 / 24);

  return () => {
    clearInterval(timer);
    ws.close();
  };
}
```

Expected packet shape:

```json
{
  "timestamp": 1715000000000,
  "yaw": 12.4,
  "pitch": -3.2,
  "roll": 1.1,
  "blink": 0.2,
  "mouth_open": 0.22,
  "brow": 0.12,
  "smile": 0.41,
  "face_landmarks": [[0.48, 0.33], [0.51, 0.34]],
  "shoulder_rotation": -2.0
}
```

The adapter also accepts the existing browser packet fields:
`blinkLeft`, `blinkRight`, `mouthOpen`, `browRaise`, `headX`, `headY`,
`shoulderX`, `shoulderY`, and compact semantic keys.

## Current Milestone Behavior

- The portrait is converted into a synthetic driving frame.
- Yaw, pitch, roll, shoulder rotation, blink, brow, smile, and mouth controls
  are drawn into that frame.
- StreamDiffusionV2 receives those frames through its existing v2v queue.
- Output frames stream back as binary websocket frames.

This deliberately does not implement PuLID, InstantID, ControlNet, identity lock,
advanced lip sync, TensorRT orchestration, or 14B-specific paths.

## Metrics

The websocket logs and sends:

- `websocket_rtt_ms`
- `queue_delay_ms`
- `adapter_latency_ms`
- `encode_latency_ms`
- `denoise_latency_ms`
- `decode_latency_ms`
- `output_fps`
- `dropped_packets`

`denoise_latency_ms` is measured at the integration boundary as stream time not
accounted for by adapter, output conversion, or websocket image encoding. It
keeps the upstream StreamV2V inference methods unchanged.
