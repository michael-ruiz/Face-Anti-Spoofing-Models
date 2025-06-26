import cv2
import mediapipe as mp

image_path = "./test_img/20130529_02_Driv_152_f .jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        # Draw landmarks
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # Draw connections
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            start_idx = connection[0]
            end_idx = connection[1]
            x1 = int(face_landmarks.landmark[start_idx].x * w)
            y1 = int(face_landmarks.landmark[start_idx].y * h)
            x2 = int(face_landmarks.landmark[end_idx].x * w)
            y2 = int(face_landmarks.landmark[end_idx].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

cv2.imshow("MediaPipe Face Mesh Landmarks & Connections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()