# app.py
import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import json
from src import config, utils

class ASLApp:
    def __init__(self, window, window_title="ASL Real-Time Translator"):
        self.window = window
        self.window.title(window_title)

        # 1. Load Model
        print("Loading model...")
        model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
        self.model = utils.load_trained_model(model_path)
        
        if self.model is None:
            print("Model not found! Please train the model first.")
            self.window.destroy()
            return

        # 2. Load Class Labels
        self.class_labels = self.load_labels()
        
        # 3. Setup Camera
        self.cap = cv2.VideoCapture(0)
        
        # GUI Layout
        self.label_video = Label(window)
        self.label_video.pack()

        self.label_pred = Label(window, text="Prediction: Waiting...", font=("Helvetica", 24))
        self.label_pred.pack(pady=20)
        
        self.label_conf = Label(window, text="Confidence: 0.0%", font=("Helvetica", 14))
        self.label_conf.pack()

        self.btn_quit = Button(window, text="Quit", command=self.close_app, width=10)
        self.btn_quit.pack(pady=10)

        # ROI Coordinates (Box where you put your hand)
        # Coordinates: (x1, y1), (x2, y2)
        self.roi_left = 100
        self.roi_top = 100
        self.roi_right = 400
        self.roi_bottom = 400
        
        self.predicting = True
        self.update_video()

    def load_labels(self):
        """Loads class mapping from JSON or falls back to default list."""
        mapping_path = os.path.join(config.BASE_DIR, 'class_indices.json')
        
        # Default list (Fallback)
        default_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            'del', 'nothing', 'space'
        ]

        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                indices = json.load(f)
                # Swap keys/values to get {0: 'A', 1: 'B'}
                return {v: k for k, v in indices.items()}
        
        return {i: label for i, label in enumerate(default_labels)}

    def update_video(self):
        """Reads frame, predicts, and updates GUI."""
        ret, frame = self.cap.read()
        
        if ret:
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # 1. Define ROI (Region of Interest)
            # This is the box where the hand must be
            cv2.rectangle(frame, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)

            # 2. Preprocess ROI for Prediction
            if self.predicting:
                # Crop the ROI
                roi_frame = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
                
                if roi_frame.size != 0:
                    # Preprocess just like in training (Resize -> RGB -> Normalize)
                    img = cv2.resize(roi_frame, config.TARGET_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Predict
                    preds = self.model.predict(img, verbose=0)
                    idx = np.argmax(preds)
                    confidence = np.max(preds)
                    
                    label = self.class_labels.get(idx, "Unknown")
                    
                    # Update Labels
                    self.label_pred.config(text=f"Prediction: {label}")
                    self.label_conf.config(text=f"Confidence: {confidence:.2%}")

            # 3. Convert frame for Tkinter Display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_image)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.label_video.imgtk = img_tk
            self.label_video.configure(image=img_tk)

        # Repeat every 10ms
        self.window.after(10, self.update_video)

    def close_app(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root, "ASL Detector")
    root.mainloop()