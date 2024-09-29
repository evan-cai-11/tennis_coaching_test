import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

class TennisCourtProjectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Court Projection Tool")

        # Video variables
        self.cap = None
        self.frame = None
        self.paused = True
        self.video_path = None

        self.resolution = (1920, 1080)

        # Point selection
        self.point = None  # The point selected by the user
        self.pts_src = []  # Points on the original image
        self.pts_src_ids = []  # IDs of the drawn points

        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the video display
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        # Create a canvas for the video
        self.canvas = tk.Canvas(self.video_frame, width=self.resolution[0], height=self.resolution[1])
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Control buttons
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack()

        self.open_button = tk.Button(self.controls_frame, text="Open Video", command=self.open_video)
        self.open_button.pack(side=tk.LEFT)

        self.play_button = tk.Button(self.controls_frame, text="Play", command=self.play_pause_video)
        self.play_button.pack(side=tk.LEFT)

        self.select_points_button = tk.Button(self.controls_frame, text="Select Court Corners", command=self.select_court_corners)
        self.select_points_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.controls_frame, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT)

        self.project_button = tk.Button(self.controls_frame, text="Project Frame", command=self.project_frame)
        self.project_button.pack(side=tk.LEFT)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.play_button.config(text="Play")
            self.paused = True
            self.play_pause_video()

    def play_video(self):
        if self.cap is not None and not self.paused:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.display_frame(frame)
            else:
                # Restart video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(30, self.play_video)

    def play_pause_video(self):
        if self.cap is None:
            return
        self.paused = not self.paused
        if not self.paused:
            self.play_button.config(text="Pause")
            self.play_video()
        else:
            self.play_button.config(text="Play")

    def display_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to fit canvas
        frame_rgb = cv2.resize(frame_rgb, (self.resolution[0], self.resolution[1]))

        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)

        # Draw selected points
        draw = ImageDraw.Draw(img)
        for pt in self.pts_src:
            x = pt[0] * self.resolution[0] / self.frame.shape[1]
            y = pt[1] * self.resolution[1] / self.frame.shape[0]
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='red')

        # Draw selected point
        if self.point is not None:
            x = self.point[0] * self.resolution[0] / self.frame.shape[1]
            y = self.point[1] * self.resolution[1] / self.frame.shape[0]
            draw.ellipse((x-5, y-5, x+5, y+5), fill='blue', outline='blue')

        # Convert to ImageTk
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk  # Keep a reference to prevent garbage collection

    def on_canvas_click(self, event):
        if self.paused and self.frame is not None:
            x = event.x * self.frame.shape[1] / self.canvas.winfo_width()
            y = event.y * self.frame.shape[0] / self.canvas.winfo_height()
            if len(self.pts_src) < 4:
                # Add court corner points
                self.pts_src.append([x, y])
                self.draw_point(event.x, event.y)
                if len(self.pts_src) == 4:
                    messagebox.showinfo("Info", "Court corners selected.")
            else:
                # Select the point to project
                self.point = np.array([x, y])
                self.display_frame(self.frame)
                messagebox.showinfo("Info", "Point selected for projection.")

    def draw_point(self, x, y, color='red'):
        point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline=color)
        self.pts_src_ids.append(point_id)

    def reset_points(self):
        self.pts_src = []
        self.point = None
        for point_id in self.pts_src_ids:
            self.canvas.delete(point_id)
        self.pts_src_ids = []
        self.display_frame(self.frame)
        messagebox.showinfo("Info", "Points have been reset.")

    def select_court_corners(self):
        if self.paused and self.frame is not None:
            messagebox.showinfo("Info", "Click on the four corners of the court in order: top-left, top-right, bottom-right, bottom-left.")

    def project_frame(self):
        if len(self.pts_src) != 4 or self.point is None:
            messagebox.showerror("Error", "Please select four court corners and a point to project.")
            return

        # Proceed with homography projection
        self.perform_projection()

    def perform_projection(self):
        # Prepare source and destination points
        pts_src = np.array(self.pts_src, dtype='float32')

        # Define standard tennis court dimensions in meters
        court_width = 10.97  # Doubles court width
        court_length = 23.77  # Court length
        scale = 20  # Adjust as needed for output size

        court_width_px = court_width * scale
        court_length_px = court_length * scale

        pts_dst = np.array([
            [0, 0],  # Top-left corner
            [court_width_px, 0],  # Top-right corner
            [court_width_px, court_length_px],  # Bottom-right corner
            [0, court_length_px]   # Bottom-left corner
        ], dtype='float32')

        # Compute homography matrix
        H, status = cv2.findHomography(pts_src, pts_dst)

        # Get the size of the original image
        h_frame, w_frame = self.frame.shape[:2]

        # Compute the corners of the original image
        corners = np.array([
            [0, 0],
            [w_frame - 1, 0],
            [w_frame - 1, h_frame - 1],
            [0, h_frame - 1]
        ], dtype='float32')

        # Project the corners onto the destination image
        projected_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)

        # Find the bounding rectangle of the projected image
        x_coords = projected_corners[:, 0]
        y_coords = projected_corners[:, 1]

        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        # Compute the translation needed to keep coordinates positive
        x_translation = -x_min if x_min < 0 else 0
        y_translation = -y_min if y_min < 0 else 0

        # Adjust the homography to include the translation
        translation_matrix = np.array([
            [1, 0, x_translation],
            [0, 1, y_translation],
            [0, 0, 1]
        ])

        H_translated = translation_matrix @ H

        # Compute the size of the output image
        output_width = int(x_max - x_min)
        output_height = int(y_max - y_min)

        # Apply warpPerspective with the adjusted homography and output size
        warped_image = cv2.warpPerspective(self.frame, H_translated, (output_width, output_height))

        # Transform the point
        point_homogeneous = np.array([self.point[0], self.point[1], 1]).reshape(3, 1)
        transformed_point = np.dot(H_translated, point_homogeneous)
        transformed_point = transformed_point / transformed_point[2]
        transformed_point = transformed_point[:2].flatten()

        # Draw the point on the warped image
        warped_image_with_point = warped_image.copy()
        cv2.circle(warped_image_with_point, (int(transformed_point[0]), int(transformed_point[1])), 10, (0, 0, 255), -1)

        # Display the result in a new window
        cv2.namedWindow('Projected Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Projected Image', warped_image_with_point)
        cv2.waitKey(0)
        cv2.destroyWindow('Projected Image')


if __name__ == "__main__":
    root = tk.Tk()
    app = TennisCourtProjectionApp(root)
    root.mainloop()
