
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the image
image_path = 'carlos.jpg'
image = cv2.imread(image_path)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect pose landmarks
results = pose.process(image_rgb)

# Check if pose landmarks were detected
if results.pose_landmarks:
    # Iterate over each landmark
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # Print the index and landmark coordinates (x, y, z)
        print(f'Landmark {idx}: name: {landmark.name} (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}, visibility: {landmark.visibility})')

# Draw landmarks on the image
mp_drawing = mp.solutions.drawing_utils
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display the result
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
