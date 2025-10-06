import os
import time
import argparse
import numpy as np
import cv2
import face_recognition
import pyttsx3

# Config
ENROLL_DIR = "enroll_images"
EMBED_FILE = "trusted_embeddings.npz"
TTS_RATE = 150
FACE_DISTANCE_THRESHOLD = 0.55
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
    import cv2
    names, embeddings = [], []
    if not os.path.isdir(enroll_dir):
        print(f"[ENROLL] Directory '{enroll_dir}' not found.")
        return names, embeddings

    files = [f for f in os.listdir(enroll_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fn in files:
        path = os.path.join(enroll_dir, fn)
        raw_name = fn.split('_')[0]
        print(f"[ENROLL] Processing {fn} for {raw_name}")

        img = cv2.imread(path)
        if img is None:
            print(f"  - Could not read {fn}, skipping.")
            continue

        # Force contiguous RGB array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.require(img, dtype=np.uint8, requirements=["C_CONTIGUOUS"])

        print(f"  - Image shape={img.shape}, dtype={img.dtype}, contiguous={img.flags['C_CONTIGUOUS']}")

        try:
            face_locs = face_recognition.face_locations(img)

        except Exception as e:
            print(f"  - Face detection error on {fn}: {e}")
            continue

        if len(face_locs) == 0:
            print(f"  - No face found in {fn}, skipping.")
            continue

        enc = face_recognition.face_encodings(img, known_face_locations=face_locs)[0]
        names.append(raw_name)
        embeddings.append(enc)

    print(f"[ENROLL] Completed enrollment for {len(names)} user(s).")
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

def compare_embedding(query_emb, known_embeddings, threshold=FACE_DISTANCE_THRESHOLD):
    """Return index of best match or None. Uses Euclidean distance."""
    if len(known_embeddings) == 0:
        return None, None
    dists = face_recognition.face_distance(known_embeddings, query_emb)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    if best_dist <= threshold:
        return int(best_idx), best_dist
    else:
        return None, best_dist

def run_recognition(camera_index=0):
    trusted_names, trusted_embeddings = load_embeddings(EMBED_FILE)
    print(f"[INFO] Loaded {len(trusted_names)} trusted embeddings.")
    if len(trusted_names) == 0:
        print("No trusted embeddings found. Run enrollment first (`--enroll_from_images` or `--interactive_enroll`).")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[RUN] Cannot open webcam.")
        return

    last_spoken = {}  # avoid repeated TTS for same person (name -> last_time)
    SPOKEN_COOLDOWN = 5.0  # seconds between greetings for same person

    print("[RUN] Starting recognition. Press 'q' in window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[RUN] Failed to read frame.")
            break

        # resize for speed
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # detect faces
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # scale face location back to original frame coords
            top *= 2; right *= 2; bottom *= 2; left *= 2

            match_idx, dist = compare_embedding(face_encoding, trusted_embeddings)
            if match_idx is not None:
                name = trusted_names[match_idx]
                label = f"{name} ({dist:.2f})"
                color = (0, 200, 0)
                # speak welcome but avoid repeating too often
                now = time.time()
                if (name not in last_spoken) or (now - last_spoken[name] > SPOKEN_COOLDOWN):
                    tts_say(f"Welcome {name}")
                    last_spoken[name] = now
            else:
                label = f"Unknown ({dist:.2f})" if dist is not None else "Unknown"
                color = (0, 0, 255)
                # Speak once per unknown person every SPOKEN_COOLDOWN seconds
                now = time.time()
                unk_key = "Unknown"
                if (unk_key not in last_spoken) or (now - last_spoken[unk_key] > SPOKEN_COOLDOWN):
                    tts_say("Hello. I do not recognize you. Please leave this room.")
                    last_spoken[unk_key] = now

            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Face Recognition (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_from_images", action="store_true", help="Create embeddings from images in enroll_images/")
    parser.add_argument("--interactive_enroll", action="store_true", help="Capture images from webcam to enroll a user")
    parser.add_argument("--recognize", action="store_true", help="Run the real-time recognition demo")
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

    elif args.recognize:
        run_recognition()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()