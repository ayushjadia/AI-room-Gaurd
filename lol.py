from PIL import Image
import numpy as np
import face_recognition

# Load image via PIL
img_pil = Image.open("enroll_images/Ayush_2.jpg").convert("RGB")  # 8-bit RGB
img = np.array(img_pil, dtype=np.uint8)              # force dtype
img = np.ascontiguousarray(img)                      # force contiguous memory

print(img.shape, img.dtype, img.flags['C_CONTIGUOUS'])

face_locs = face_recognition.face_locations(img)
print(face_locs)
