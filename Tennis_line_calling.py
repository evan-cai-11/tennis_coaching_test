import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import mediapipe as mp
from tkinter import ttk
from PIL import Image, ImageTk
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Racket Detection")
        self.root.geometry("800x600")

        # Initialize video variables
        self.cap = None
        self.video_path = None
        self.paused = True
        self.frame_num = 0
        self.total_frames = 0
        self.canvas_width = 640
        self.canvas_height = 480

        # Add a list to store previous ball positions
        self.ball_positions = []
        self.max_positions = 10  # Maximum number of positions to store
        self.skip_counter = 0
        self.max_skip = 3

        # Store the last processed image
        self.imgtk = None

        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Configure grid
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Button to select video
        self.select_button = tk.Button(self.root, text="Select Video", command=self.open_file)
        self.select_button.grid(row=0, column=0, pady=10)

        # Create a canvas for video display
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.grid(row=1, column=0, sticky="nsew")  # Expand canvas

        # Bind resize event to adjust video display area
        self.root.bind("<Configure>", self.resize_canvas)

        # Controls frame
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=2, column=0, pady=10, sticky="ew")
        self.controls_frame.columnconfigure(0, weight=1)

        # Progress bar for video playback
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(self.controls_frame, from_=0, to=100, orient="horizontal",
                                      variable=self.progress_var, command=self.seek_video)
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        # Play/Pause button
        self.play_button = tk.Button(self.controls_frame, text="Play", command=self.toggle_playback)
        self.play_button.grid(row=0, column=1, padx=10)

    def open_file(self):
        # Open a file dialog to select a video file
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not self.video_path:
            return

        # Open the selected video file with OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.paused = False

        # Update the play button and start processing the video
        self.play_button.config(text="Pause")
        self.play_video()

    def play_video(self):
        if not self.cap:
            return

        if not self.paused:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video if it ends
                self.frame_num = 0
                return

            # Update the progress bar
            self.frame_num += 1
            self.progress_var.set((self.frame_num / self.total_frames) * 100)
            self.process_frame(frame)
        else:
            # If paused, use the last processed image
            pass  # No need to process frame when paused

        # Update the canvas with the new frame
        if self.imgtk:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        # Call play_video again after a delay
        self.root.after(10, self.play_video)

    def process_frame(self, frame):
        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run object detection using YOLOv7 on RGB image
        results = model(image_rgb)

        ball_detected = False
        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()

            # Filter out low-confidence detections (confidence > 0.5)
            if conf > 0.5 and model.names[int(cls)] == "sports ball":
                ball_detected = True
                # Calculate center of the bounding box
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                # Add current position to the list
                self.ball_positions.append((center_x, center_y))
                
                # Limit the number of stored positions
                if len(self.ball_positions) > self.max_positions:
                    self.ball_positions.pop(0)

                # Draw bounding box
                cv2.rectangle(image_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Display class label
                label = f'{model.names[int(cls)]}: {conf:.2f}'
                cv2.putText(image_rgb, label, (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw lines connecting ball positions
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                cv2.line(image_rgb, self.ball_positions[i-1], self.ball_positions[i], (0, 255, 0), 2)

        # If no ball is detected in this frame, increment the skip counter
        if not ball_detected:
            self.skip_counter += 1
            if self.skip_counter > self.max_skip:  # Allow skipping up to 3 frames
                self.ball_positions.clear()
                self.skip_counter = 0
        else:
            self.skip_counter = 0  # Reset skip counter when ball is detected

        # Run MediaPipe Pose Detection on RGB image
        #poses = pose.process(image_rgb)

        # Draw the pose on the RGB image
        #if poses.pose_landmarks:
        #    mp_drawing.draw_landmarks(image_rgb, poses.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize the RGB image to fit within the canvas size while maintaining aspect ratio
        image_rgb = self.resize_frame_to_canvas(image_rgb)

        # Convert RGB image to ImageTk for displaying in Tkinter Canvas
        img = Image.fromarray(image_rgb)
        self.imgtk = ImageTk.PhotoImage(image=img)  # Store reference to avoid garbage collection


    def toggle_playback(self):
        self.paused = not self.paused
        if self.paused:
            self.play_button.config(text="Play")
        else:
            self.play_button.config(text="Pause")

    def seek_video(self, value):
        # Seek video to a specific frame based on the progress bar value
        if self.cap:
            frame_number = int(float(value) / 100 * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_num = frame_number
            if self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

    def resize_canvas(self, event):
        # Update canvas size when the window is resized
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)

    def resize_frame_to_canvas(self, frame):
        # Get the aspect ratio of the frame and the canvas
        frame_height, frame_width = frame.shape[:2]
        canvas_ratio = self.canvas_width / self.canvas_height
        frame_ratio = frame_width / frame_height

        # Resize the frame to fit within the canvas while maintaining aspect ratio
        if frame_ratio > canvas_ratio:
            new_width = self.canvas_width
            new_height = int(new_width / frame_ratio)
        else:
            new_height = self.canvas_height
            new_width = int(new_height * frame_ratio)

        # Resize the frame using OpenCV
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame

# Create the main window and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
