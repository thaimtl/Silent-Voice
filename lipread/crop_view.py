"""E1 debug view: the camera frame (mirrored for display only) next to the 96x96 crop the model reads, with face +
crop ms per frame and the top blendshapes. A window when there is a display, else (or with --out) an MP4.

    venv-live/bin/python -m lipread.crop_view                      # webcam 0, window (q quits)
    venv-live/bin/python -m lipread.crop_view --replay third_party/lipflow/samples/2016-03-12.mov --start 20.4 \\
        --end 28.1 --out runs/lipread/crop_view.mp4
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np

import lipread
from lipread.crop import BLEND_NAMES, LiveCropper
from lipread.run import fit
from lipread.source import CameraSource, ReplaySource
from lipread.types import FPS

SCALE = 4


def panel(frame, crop, ms):
    """Side-by-side BGR image: mirrored frame scaled to 384 high | crop x4 | text."""
    h = CROP_PX = 96 * SCALE
    img = cv2.flip(frame.image, 1)
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    roi = cv2.cvtColor(cv2.resize(crop.roi96, (CROP_PX, CROP_PX), interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)
    out = np.hstack([img, roi])
    lines = [f't {crop.t:8.2f}  face {"ok" if crop.face_ok else "--"}  crop {ms:5.1f} ms',
             f'mouth_open {crop.mouth_open:.3f}']
    if crop.blend is not None and BLEND_NAMES:
        top = np.argsort(crop.blend)[::-1][:4]
        lines += [f'{BLEND_NAMES[i]} {crop.blend[i]:.2f}' for i in top]
    for j, s in enumerate(lines):
        cv2.putText(out, s, (8, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def paired(src, cropper, process_size=(1280, 720)):
    """(frame, crop) for every frame of src, shrunk to process_size first as lipread.run does. The cropper returns one
    crop per frame, in order and k frames late, so a FIFO pairs them (frame times repeat when the 25 fps grid repeats a
    frame of a slower video or camera)."""
    pending = deque()
    for frame in src.frames():
        frame = fit(frame, process_size)
        pending.append(frame)
        for crop in cropper.push(frame):
            yield pending.popleft(), crop
    for crop in cropper.flush():
        yield pending.popleft(), crop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replay', default=None, help='video file instead of the webcam')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--out', default=None, help='write an MP4 instead of (or when there is no) window')
    args = ap.parse_args(argv)

    src = (ReplaySource(args.replay, args.start, args.end, realtime=args.out is None) if args.replay
           else CameraSource(args.cam))
    cropper = LiveCropper(os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task'))
    has_display = sys.platform in ('win32', 'darwin') or bool(os.environ.get('DISPLAY') or
                                                              os.environ.get('WAYLAND_DISPLAY'))
    show = args.out is None and has_display
    out_path = args.out or (None if show else os.path.join(lipread.WORK, 'runs', 'lipread', 'crop_view.mp4'))
    writer, ms = None, []
    try:
        for frame, crop in paired(src, cropper):
            ms.append(cropper.times[-1])
            img = panel(frame, crop, cropper.times[-1])
            if show:
                cv2.imshow('crop (q quits)', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            else:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cropper.close()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if ms:
        print(f'{len(ms)} crops, face + crop ms/frame: median {np.median(ms):.2f}, p95 {np.percentile(ms, 95):.2f}'
              + (f', written to {out_path}' if out_path else ''))


if __name__ == '__main__':
    main()
