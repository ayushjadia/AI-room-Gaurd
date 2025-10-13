import os
import time
import argparse
import numpy as np
import cv2
import face_recognition
import pyttsx3
from config import *


# lower threshold => stricter matching; 0.4-0.6 typical for face_recognition embeddings

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)

def tts_say(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

def load_embeddings(npz_path):
    """Load embeddings and names from npz file."""
    if not os.path.exists(npz_path):
        return [], []
    data = np.load(npz_path, allow_pickle=True)
    names = data["names"].tolist()
    embeddings = data["embeddings"]
    return names, embeddings

def save_embeddings(npz_path, names, embeddings):
    """Save embeddings and names to npz."""
    np.savez(npz_path, names=np.array(names, dtype=object), embeddings=np.array(embeddings))
    print(f"[I/O] Saved {len(names)} trusted embeddings to {npz_path}")

def enroll_from_images(enroll_dir=ENROLL_DIR):
    """Load images in enroll_dir, compute embeddings. Filenames should be Name_#.jpg"""
    names = []
    embeddings = []
    if not os.path.isdir(enroll_dir):
        print(f"[ENROLL] Directory '{enroll_dir}' not found. Create it and add images named like Alice_1.jpg")
        return names, embeddings

    files = [f for f in os.listdir(enroll_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fn in files:
        path = os.path.join(enroll_dir, fn)
        # name parsing: anything before the first underscore
        raw_name = fn.split('_')[0]
        print(f"[ENROLL] Processing {fn} for {raw_name}")
        img = face_recognition.load_image_file(path)
        face_locs = face_recognition.face_locations(img)
        if len(face_locs) == 0:
            print(f"  - No face found in {fn}, skipping.")
            continue
        # Use first face found
        enc = face_recognition.face_encodings(img, known_face_locations=face_locs)[0]
        names.append(raw_name)
        embeddings.append(enc)
    return names, embeddings

def interactive_enroll(camera_index=0, num_images=3, resize_width=320):
    """Capture images via webcam for a new user and compute average embedding."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ENROLL] Cannot open webcam for enrollment.")
        return [], []

    user_name = input("Enter name for enrollment (no spaces): ").strip()
    if not user_name:
        print("Invalid name.")
        cap.release()
        return [], []

    captured_embeddings = []
    print(f"[ENROLL] Capturing {num_images} images for user '{user_name}'. Press SPACE to capture each image.")
    cv2.namedWindow("Enrollment (press q to cancel)")
    while len(captured_embeddings) < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ENROLL] Frame capture failed.")
            break
        # show scaled
        small = cv2.resize(frame, (resize_width, int(frame.shape[0]*resize_width/frame.shape[1])))
        cv2.imshow("Enrollment (press q to cancel)", small)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE to capture
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 0:
                print(" No face detected. Try again.")
                continue
            encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
            # take first face
            captured_embeddings.append(encs[0])
            print(f" Captured {len(captured_embeddings)}/{num_images}")
            time.sleep(0.5)
        elif key == ord('q'):
            print("Enrollment canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_embeddings) == 0:
        return [], []

    # Average embeddings to get a stable template
    avg_embedding = np.mean(np.array(captured_embeddings), axis=0)
    return [user_name], [avg_embedding]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_from_images", action="store_true", help="Create embeddings from images in enroll_images/")
    parser.add_argument("--interactive_enroll", action="store_true", help="Capture images from webcam to enroll a user")
    args = parser.parse_args()

    if args.enroll_from_images:
        names, embeddings = enroll_from_images()
        if names:
            existing_names, existing_embeddings = load_embeddings(EMBED_FILE)
            combined_names = existing_names + names
            combined_embeddings = list(existing_embeddings) + embeddings if len(existing_embeddings)>0 else embeddings
            save_embeddings(EMBED_FILE, combined_names, combined_embeddings)
        else:
            print("No new enrollments found.")

    elif args.interactive_enroll:
        names, embeddings = interactive_enroll()
        if names:
            existing_names, existing_embeddings = load_embeddings(EMBED_FILE)
            combined_names = existing_names + names
            combined_embeddings = list(existing_embeddings) + embeddings if len(existing_embeddings)>0 else embeddings
            save_embeddings(EMBED_FILE, combined_names, combined_embeddings)
        else:
            print("Interactive enrollment produced no embeddings.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()