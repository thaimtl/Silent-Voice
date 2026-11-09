"""HSEmotion (enet_b0_8_best_vgaf, Apache-2.0) on the face crop, plus a head fitted by lipread/emotion_fit.py."""
import hashlib
import logging
import os

import numpy as np

import lipread

MODEL_FILE = 'enet_b0_8_best_vgaf.onnx'
MODEL_PATH = os.path.join(lipread.LIPFLOW_MODELS, 'hsemotion', MODEL_FILE)
MODEL_SHA256 = 'fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898'
HEAD_PATH = os.path.join(lipread.WORK, 'data', 'emotion', 'head.npz')

log = logging.getLogger(__name__)
CLASSES = ('Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise')
# no head: AffectNet class -> bus label (the other three classes are dropped)
GENERIC = {'Neutral': 'neutral', 'Happiness': 'happy', 'Sadness': 'sad', 'Anger': 'angry', 'Surprise': 'surprised'}
SIZE = 224
MARGIN = 1.15
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N_FEATURES = len(CLASSES) + 52


def face_box(pts, w, h, margin=MARGIN):
    """Landmarks (N, 2) in pixels -> square (x0, y0, x1, y1) clipped to the image, or None if it is too small."""
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * margin / 2
    box = (max(0, int(cx - s)), max(0, int(cy - s)), min(w, int(cx + s)), min(h, int(cy + s)))
    return box if box[2] - box[0] >= 32 and box[3] - box[1] >= 32 else None


def preprocess(face_rgb):
    import cv2
    x = cv2.resize(face_rgb, (SIZE, SIZE)).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]


class ExpressionNet:
    def __init__(self, path=MODEL_PATH, threads=2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads           # leave the cores to the lip reader
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])
        self.input = self.sess.get_inputs()[0].name

    @staticmethod
    def available(path=MODEL_PATH, sha256=MODEL_SHA256):
        """onnxruntime importable and the model file present with the pinned checksum."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        if not os.path.isfile(path):
            return False
        if sha256 is None:
            return True
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest() == sha256

    def logits(self, image_bgr, pts):
        import cv2
        h, w = image_bgr.shape[:2]
        box = face_box(np.asarray(pts, np.float32), w, h)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        face = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        return self.sess.run(None, {self.input: preprocess(face)})[0][0, :len(CLASSES)].astype(np.float32)


LOGITS = tuple(range(len(CLASSES)))      # the HSEmotion scores alone: what carries across sessions (emotion_fit)


class Head:
    """Standardise + logistic regression in numpy, with the smoothing it was tuned with. cols: which of the
    N_FEATURES [8 logits, 52 blendshapes] it reads (an older head file without them reads all)."""

    def __init__(self, labels, mean, scale, coef, intercept, margin, hold_s, tau_s, info='', cols=None):
        self.labels = tuple(str(x) for x in labels)
        self.cols = np.arange(N_FEATURES) if cols is None else np.asarray(cols, np.int64)
        self.mean, self.scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        self.coef, self.intercept = np.asarray(coef, np.float32), np.asarray(intercept, np.float32)
        self.margin, self.hold_s, self.tau_s = float(margin), float(hold_s), float(tau_s)
        self.info = str(info)
        n = len(self.cols)
        if self.coef.shape != (len(self.labels), n) or self.mean.shape != (n,) or self.scale.shape != (n,) \
                or 'neutral' not in self.labels or not ((0 <= self.cols) & (self.cols < N_FEATURES)).all():
            raise ValueError(f'head: coef {self.coef.shape} for labels {self.labels}, features {self.cols.tolist()}')

    def predict(self, x):
        """x: (N, N_FEATURES) or (N_FEATURES,), baseline already subtracted -> (N, len(labels))."""
        x = np.atleast_2d(x)[:, self.cols]
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, labels=np.array(self.labels), mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=self.intercept, smoothing=np.array([self.margin, self.hold_s, self.tau_s]),
                 info=np.array(self.info), cols=self.cols)

    @classmethod
    def load(cls, path=HEAD_PATH):
        z = np.load(path, allow_pickle=False)
        if 'cols' not in z:
            log.warning('%s was fitted against the first-seconds baseline; refit it: python -m lipread.emotion_fit',
                        path)
        margin, hold_s, tau_s = z['smoothing']
        return cls(z['labels'], z['mean'], z['scale'], z['coef'], z['intercept'], margin, hold_s, tau_s,
                   z['info'].item() if 'info' in z else '', z['cols'] if 'cols' in z else None)


def generic_probs(logits):
    """No head: softmax over the five AffectNet classes that have a bus label, in GENERIC's order."""
    z = np.asarray(logits, np.float32)[[CLASSES.index(c) for c in GENERIC]]
    z = np.exp(z - z.max())
    return z / z.sum()
