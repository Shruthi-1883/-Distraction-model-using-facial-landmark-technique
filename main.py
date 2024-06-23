import cv2
import dlib
import numpy as np
import winsound  # For playing sound (Windows-specific)

# Load pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect distraction
def detect_distraction(frame, eyes_closed_threshold=0.2, phone_in_hand=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract relevant facial landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Calculate eye aspect ratio (EAR) for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average EAR of both eyes
        ear = (left_ear + right_ear) / 2

        # Check for distraction based on EAR threshold and phone usage
        if ear < eyes_closed_threshold or phone_in_hand:
            return True  # Distraction detected

    return False  # No distraction detected

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play sound when distracted
def play_sound():
    # Beep sound (frequency in Hz, duration in milliseconds)
    winsound.Beep(1000, 500)

# Main function
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect distraction
        distracted = detect_distraction(frame)

        # Display result and play sound if distracted
        if distracted:
            cv2.putText(frame, "DISTRACTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            play_sound()  # Play sound if distracted
        else:
            cv2.putText(frame, "FOCUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Distraction Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

