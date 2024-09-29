import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load an image
image = cv2.imread('carlos.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find the pose
results = pose.process(image_rgb)

# Draw the pose on the image
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Show the image
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
