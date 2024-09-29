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
        self.root.geometry("1600x1200")

        # Initialize video variables
        self.cap = None
        self.video_path = None
        self.paused = True
        self.frame_num = 0
        self.total_frames = 0
        self.min_canvas_width = 800
        self.min_canvas_height = 600
        self.initial_canvas_width = 1600
        self.initial_canvas_height = 1200
        self.canvas_width = self.initial_canvas_width
        self.canvas_height = self.initial_canvas_height

        # Add a list to store previous ball positions
        self.ball_positions = []
        self.max_positions = 10  # Maximum number of positions to store
        self.skip_counter = 0
        self.max_skip = 3
        self.contact_overlap = 0.8

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

        # Create a canvas for video display with fixed size
        self.canvas = tk.Canvas(self.root, bg="black", width=self.initial_canvas_width, height=self.initial_canvas_height)
        self.canvas.grid(row=1, column=0, sticky="nsew")  # Expand canvas

        # Remove the bind for resize event
        self.root.bind("<Configure>", self.on_window_resize)

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

            # Ensure canvas size is correct before processing the frame
            #self.update_canvas_size()

            self.process_frame(frame)


        # Update the canvas with the new frame
        if self.imgtk:
            self.canvas.delete("all")  # Clear previous image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        # Call play_video again after a delay
        self.root.after(5, self.play_video)

    def detect_contact(self, ball_positions, racket_positions):
        if ball_positions and racket_positions:
            ball_xmin, ball_ymin, ball_xmax, ball_ymax = ball_positions
            racket_xmin, racket_ymin, racket_xmax, racket_ymax = racket_positions

            # Calculate the overlap between the ball and the racket
            overlap_x = min(ball_xmax, racket_xmax) - max(ball_xmin, racket_xmin)
            overlap_y = min(ball_ymax, racket_ymax) - max(ball_ymin, racket_ymin)
            overlap_area = max(0, overlap_x) * max(0, overlap_y)

            # Calculate the area of the ball
            ball_area = (ball_xmax - ball_xmin) * (ball_ymax - ball_ymin)

            # Calculate the intersection over union (IoU)
            iou = overlap_area / ball_area
            return iou > self.contact_overlap

        return False

    def process_frame(self, frame):
        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run object detection using YOLOv7 on RGB image
        results = model(image_rgb)

        ball_positions = []
        racket_positions = []
        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()

            # Filter out low-confidence detections (confidence > 0.5)
            if conf > 0.5 and model.names[int(cls)] in ["tennis racket", "sports ball"]:

                if model.names[int(cls)] == "sports ball":
                    ball_positions = [xmin, ymin, xmax, ymax]
                elif model.names[int(cls)] == "tennis racket":
                    racket_positions = [xmin, ymin, xmax, ymax] 

                # Draw bounding box
                cv2.rectangle(image_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

                # Display class label
                label = f'{model.names[int(cls)]}: {conf:.2f}'
                cv2.putText(image_rgb, label, (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.detect_contact(ball_positions, racket_positions):
            print("Contact detected")
            self.paused = True
            self.play_button.config(text="Play")

        # Run MediaPipe Pose Detection on RGB image
        poses = pose.process(image_rgb)

        # Draw the pose on the RGB image
        if poses.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, poses.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
        
        # Force update of the window to ensure all widget sizes are current
        self.root.update_idletasks()

    def seek_video(self, value):
        if self.cap:
            frame_number = int(float(value) / 100 * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_num = frame_number
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

    def on_window_resize(self, event):
        # Only resize if the event is for the root window, not child widgets
        if event.widget == self.root:
            # Ensure minimum size
            new_width = max(event.width, self.min_canvas_width)
            new_height = max(event.height - 100, self.min_canvas_height)  # Subtract space for controls

            if new_width != self.canvas_width or new_height != self.canvas_height:
                self.canvas_width = new_width
                self.canvas_height = new_height
                self.canvas.config(width=self.canvas_width, height=self.canvas_height)
                print(f"Canvas resized to: {self.canvas_width}x{self.canvas_height}")


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
