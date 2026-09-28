import cv2
import numpy as np
from types import SimpleNamespace
from xray_dip_app import XRayDIPApp

app = XRayDIPApp.__new__(XRayDIPApp)
app.denoise_var = SimpleNamespace(get=lambda: "Median")
app.enhance_var = SimpleNamespace(get=lambda: "CLAHE")
app.segment_var = SimpleNamespace(get=lambda: "Otsu threshold")
app.morph_var = SimpleNamespace(get=lambda: "Opening + closing")
app.kernel_var = SimpleNamespace(get=lambda: 5)
app.gamma_var = SimpleNamespace(get=lambda: 1.0)

image = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(image, (128, 128), 70, 180, -1)
cv2.rectangle(image, (40, 40), (80, 210), 90, -1)
preprocessed = app.preprocess(image)
enhanced = app.enhance(preprocessed)
mask = app.postprocess(app.segment(enhanced))
assert preprocessed.shape == image.shape
assert enhanced.dtype == np.uint8
assert mask.shape == image.shape
assert set(np.unique(mask)).issubset({0, 255})
print("pipeline smoke test passed", image.shape, int(mask.sum() > 0))
