# Vision model boundary

## Existing server-side assets

- `models/yolo26n.pt`: Ultralytics object-detection checkpoint used by `app/services/vision.py`.
- `models/yolo11n-pose.pt`: Ultralytics pose checkpoint.
- `app/services/face_recognition.py`: InsightFace `buffalo_l` pipeline using CPUExecutionProvider.
- `app/services/face_enrollment.py`: server-side face enrollment and multi-frame sample selection.

## Mobile integration boundary

These Python checkpoints and InsightFace dependencies are backend assets. They are not directly loadable from the Expo React Native app. For true offline mobile inference, export/validate compatible mobile model formats and integrate them with a supported native runtime in the mobile repository.

## Privacy requirement

The mobile face gallery must remain device-local. Do not call the server-side `/api/faces/*` endpoints for local enrollment or identity matching. Server-side endpoints may remain available for other explicitly opted-in workflows, but must not receive biometric samples from the offline mode.

## Validation required

- Validate exported model parity against the source checkpoints.
- Test detection and pose accuracy at intended camera resolutions and distances.
- Benchmark target-device latency, memory and thermals.
- Verify offline behavior with network disabled.
- Keep face identity matching separate from pose estimation; posture is not an identity signal.
