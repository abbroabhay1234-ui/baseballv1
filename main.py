import cv2
import numpy as np
import time
import os
import collections
from datetime import datetime

# Import our previous modules
from camera_sync import DualCameraStream
from ball_tracker import BallTracker

class DigitalUmpire:
    def __init__(self, cam1_src=0, cam2_src=1):
        """
        Main logic class for the 3D Digital Umpire.
        Integrates synchronized camera feeds, YOLOv8 ball tracking, and homography evaluation.
        """
        print("Initializing Dual Cameras...")
        self.stream = DualCameraStream(src1="http://192.168.129.153:8080/video", src2="http://192.168.129.70:8080/video")
        
        # 1. Load Homography Matrix for Camera 2 Strike Zone
        if not os.path.exists("homography.npy"):
            print("WARNING: homography.npy not found! Please run Phase 2 calibration FIRST.")
            # Fallback to an identity matrix for testing without throwing errors
            self.H = np.eye(3, dtype=np.float32)
        else:
            self.H = np.load("homography.npy")
            print("Successfully loaded Homography Matrix.")
            
        # These would ideally be fetched from the Phase 2 GUI. Hardcoded here as defaults.
        self.sz_x, self.sz_y, self.sz_w, self.sz_h = 200, 300, 400, 600
        
        # Calculate the warped skewed polygon that visually represents the strike zone in Camera 2
        canonical_pts = np.array([
            [self.sz_x, self.sz_y],
            [self.sz_x + self.sz_w, self.sz_y],
            [self.sz_x + self.sz_w, self.sz_y + self.sz_h],
            [self.sz_x, self.sz_y + self.sz_h]
        ], dtype="float32").reshape(-1, 1, 2)
        self.strike_zone_poly = np.int32(cv2.perspectiveTransform(canonical_pts, self.H))

        # 2. Initialize Machine Learning Ball Trackers for BOTH cameras
        # Optimization: We increased imgsz to 640 and lowered confidence to 0.1 
        # because a fast-moving baseball is too small and blurry to detect at 320x320.
        self.tracker_cam1 = BallTracker(model_path="yolov8n.pt", imgsz=640, conf_threshold=0.1)
        self.tracker_cam2 = BallTracker(model_path="yolov8n.pt", imgsz=640, conf_threshold=0.1)
        
        # 3. Game State / Trigger Logic Variables
        # This X-coordinate on Camera 1 simulates the "front plane" of home plate
        self.trigger_line_x = 640  
        
        self.pitch_result = None
        self.result_frames_left = 0
        
        # New API and Replay Recording states
        self.latest_status = "Waiting for pitch..."
        self.frame_buffer = collections.deque(maxlen=90) # Store last 3 seconds
        self.recording_frames_left = 0
        self.pitch_count = 0
        
    def evaluate_pitch(self, ball_center_cam2):
        """
        Phase 4: The Evaluation Function.
        Checks if the provided (x,y) point is inside the 4-point Skewed Strike Zone Polygon.
        Returns "STRIKE" if inside, "BALL" if outside.
        """
        # pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
        dist = cv2.pointPolygonTest(self.strike_zone_poly, ball_center_cam2, measureDist=False)
        if dist >= 0:
            return "STRIKE"
        else:
            return "BALL"

    def process_frame(self):
        """Processes a single pair of frames and returns (frame1, frame2). Returns (None, None) if no frames available."""
        # Poll the absolute fastest, most synchronized frame pair
        frame1, frame2, t1, t2 = self.stream.get_paired_frames()
        if frame1 is None or frame2 is None:
            return None, None
            
        # Keep a rolling buffer of frames for our highlight replay video
        self.frame_buffer.append(frame2.copy())
            
        trigger_active = False
        
        # ==========================================
        # --- CAMERA 1 LOGIC (Side View - Trigger) ---
        # ==========================================
        # Draw the trigger plane indicating the front of home plate
        cv2.line(frame1, (self.trigger_line_x, 0), (self.trigger_line_x, frame1.shape[0]), (0, 255, 255), 2)
        cv2.putText(frame1, "Trigger Plane (Plate Front)", (self.trigger_line_x + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        objs1, traj1, _ = self.tracker_cam1.detect_and_track(frame1)
        
        for obj_id, centroid in objs1.items():
            cv2.circle(frame1, tuple(centroid), 5, (255, 0, 0), -1)
            
            # Logic: If ball passes the trigger line (e.g. traveling right to left contextually)
            # Note: We apply a small window (e.g., 50px buffer) so it only triggers ONCE as it crosses.
            if self.trigger_line_x - 50 < centroid[0] <= self.trigger_line_x: 
                trigger_active = True
                cv2.putText(frame1, "TRIGGERED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # ===============================================
        # --- CAMERA 2 LOGIC (Judge View - Evaluation) ---
        # ===============================================
        objs2, traj2, _ = self.tracker_cam2.detect_and_track(frame2)
        
        # Visual Feedback: Draw the Trail of the baseball using tracking history
        for obj_id, past_centers in traj2.items():
            # Draw a line connecting the last 15 trajectory points
            max_trail = min(len(past_centers), 15)
            for i in range(1, max_trail):
                # Reverse list conceptually to get the tail
                pt1 = past_centers[-i]
                pt2 = past_centers[-(i + 1)]
                cv2.line(frame2, tuple(pt1), tuple(pt2), (0, 165, 255), 2)
        
        ball_center_c2 = None
        # Get current active ball (simplifying to assume 1 ball in frame)
        for obj_id, centroid in objs2.items():
            ball_center_c2 = tuple(centroid)
            cv2.circle(frame2, ball_center_c2, 6, (0, 165, 255), -1)
            break 

        # THE CORE INTEGRATION LOGIC:
        # IF Trigger == True AND we see the ball in Camera 2
        if trigger_active and ball_center_c2 is not None:
            # Evaluate position relative to Strike Zone Polygon
            self.pitch_result = self.evaluate_pitch(ball_center_c2)
            self.result_frames_left = 60 # Keep the result on screen for 60 frames (~2 secs)
            self.latest_status = f"PITCH DETECTED: {self.pitch_result}!"
            
            # Start holding the aftermath for 60 frames before dumping highlight replay
            self.recording_frames_left = 60
            self.pitch_count += 1
        
        # --- APPLY VISUAL FEEDBACK ON OUTPUT ---
        box_color = (255, 255, 255) # Default White Zone Outline
        
        if self.result_frames_left > 0:
            if self.pitch_result == "STRIKE":
                box_color = (0, 255, 0) # Flash Zone Green for Strike
                cv2.putText(frame2, "STRIKE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            elif self.pitch_result == "BALL":
                box_color = (0, 0, 255) # Flash Zone Red for Ball
                cv2.putText(frame2, "BALL!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            self.result_frames_left -= 1
            
        # Draw the properly anchored, skewed Strike Zone Polygon!
        cv2.polylines(frame2, [self.strike_zone_poly], isClosed=True, color=box_color, thickness=3)

        # Handle recording countdown
        if self.recording_frames_left > 0:
            self.recording_frames_left -= 1
            if self.recording_frames_left == 0:
                self._save_replay()
        elif self.result_frames_left == 0:
            self.latest_status = "Waiting for pitch..."

        return frame1, frame2

    def _save_replay(self):
        filename = f"pitch_highlight_{self.pitch_count}_{datetime.now().strftime('%H%M%S')}.mp4"
        print(f"\n[RECORDING] Saving pitch highlight video: {filename}...")
        
        if len(self.frame_buffer) > 0:
            height, width, layers = self.frame_buffer[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            for f in self.frame_buffer:
                out.write(f)
            out.release()
            print("[RECORDING] Replay saved successfully!\n")

    def run(self):
        """Starts the main evaluation loop (Desktop GUI)."""
        self.stream.start()
        time.sleep(2.0) # Camera warmup
        print("Cameras Active. Press 'q' to quit.")
        
        try:
            while True:
                f1, f2 = self.process_frame()
                if f1 is None:
                    continue

                # --- RENDER BOTH CAMERA OUTPUTS ---
                cv2.imshow("Camera 1 - Trigger View (Z-Axis)", f1)
                cv2.imshow("Camera 2 - Judge View (X-Y Axis)", f2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        except KeyboardInterrupt:
            print("Manually Interrupted.")
        finally:
            self.stream.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # To run this in reality, replace 0 and 1 with actual hardware capture card indices or video file paths.
    # Example: umpire = DigitalUmpire(cam1_src="side_view.mp4", cam2_src="behind_umpire.mp4")
    print("--- 3D Digital Umpire Demo ---")
    umpire = DigitalUmpire(cam1_src=0, cam2_src=1)
    umpire.run()
