import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import OrderedDict

class CentroidTracker:
    """
    A simple but effective tracking algorithm that correlates bounding boxes
    across frames using Euclidean distance. Helps reduce false positives
    and maintains a clear historical trajectory of the ball.
    """
    def __init__(self, max_disappeared=5, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()       # Active objects {id: centroid}
        self.disappeared = OrderedDict()   # Frame count an object has vanished
        self.trajectories = OrderedDict()  # Historical trajectory {id: [centroids]}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        # We intentionally keep the trajectory in self.trajectories for evaluation later

    def update(self, rects):
        # If no bounding boxes are found, tick the "disappeared" counter for everything
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.trajectories

        # Calculate centroids for current bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between existing objects and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Match current objects to new inputs by finding smallest distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                # If distance is too far, do not associate them
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col])
                
                used_rows.add(row)
                used_cols.add(col)

            # Untracked existing objects vanish this frame
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Unmatched new inputs become new objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects, self.trajectories


class BallTracker:
    """
    Wraps YOLOv8 inference and centroid tracking.
    """
    def __init__(self, model_path="yolov8n.pt", imgsz=320, conf_threshold=0.25):
        """
        For raw real-time speed, we limit the inference resolution (`imgsz`).
        In production, a custom YOLO model trained *only* on baseballs is highly recommended.
        """
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        # 'sports ball' is class ID 32 in standard COCO dataset for YOLOv8
        self.ball_class_id = 32 
        
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=150)
    
    def detect_and_track(self, frame):
        """
        Runs YOLO inference on the frame, returning object coordinates and tracking history.
        """
        # Run inference. Verbose=False keeps terminal clean.
        # We exclusively filter for sports balls (classes=[32]).
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold, 
                               classes=[self.ball_class_id], verbose=False)
        
        rects = []
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            rects.append((int(x1), int(y1), int(x2), int(y2)))
        
        # Update tracker with current bounding boxes to maintain trajectory
        objects, trajectories = self.tracker.update(rects)
        
        return objects, trajectories, rects

if __name__ == "__main__":
    print("Testing Phase 3 ML Pipeline Layout...")
    
    # Initialize the Tracker
    # It will automatically download 'yolov8n.pt' from Ultralytics if not found.
    print("Setting up YOLO and Centroid Tracker... (first run may download model file)")
    detector = BallTracker(model_path="yolov8n.pt", imgsz=320)
    
    print()
    print("Phase 3 complete! The `BallTracker` is ready to be integrated.")
