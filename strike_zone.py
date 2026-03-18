import cv2
import numpy as np

class StrikeZoneCalibrator:
    """
    Handles the 4-point click calibration for Camera 2.
    Calculates the Homography matrix to map from an idealized 2D canonical space
    to the skewed camera view.
    """
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points = []
        self.homography = None

    def click_event(self, event, x, y, flags, param):
        """Mouse callback to capture 4 corners of the strike zone face."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Calibration - Click 4 Corners", self.frame)

            if len(self.points) == 4:
                # 1. Source points: The 4 clicked points in the skewed camera view
                pts_camera = np.array(self.points, dtype="float32")
                
                # 2. Destination points: The IDEAL 2D front-facing canonical strike zone.
                # E.g., a standardized 400x600 rectangle.
                pts_canonical = np.array([
                    [0, 0],         # Top-Left
                    [400, 0],       # Top-Right
                    [400, 600],     # Bottom-Right
                    [0, 600]        # Bottom-Left
                ], dtype="float32")
                
                # Calculate the matrix to map canonical 2D BACK to the skewed camera view
                # We use canonical -> camera so we can adjust an ideal rectangle in canonical coordinates,
                # then warp it to draw correctly on the angled camera feed!
                self.homography, _ = cv2.findHomography(pts_canonical, pts_camera)
                print("Homography Matrix Calibrated Successfully!")

    def calibrate(self):
        cv2.imshow("Calibration - Click 4 Corners", self.frame)
        cv2.setMouseCallback("Calibration - Click 4 Corners", self.click_event)
        print("Please click the 4 corners of the Strike Zone on the video feed.")
        print("Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
        print("Press any key after clicking all 4 points to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.homography


class StrikeZoneGUI:
    """
    Applies the Homography matrix to provide a user-adjustable 2D strike zone.
    The user adjusts intuitive 2D parameters (Width, Height, position), and the GUI
    maps it to the skewed perspective of the plate accurately.
    """
    def __init__(self, homography):
        self.H = homography
        self.window_name = "Camera 2 - Adjustable Strike Zone"

    def _on_trackbar(self, val):
        pass

    def run_gui(self, frame):
        cv2.namedWindow(self.window_name)
        # Trackbars to intuitively move and resize a 2D bounding box
        cv2.createTrackbar("X Pos", self.window_name, 200, 800, self._on_trackbar)
        cv2.createTrackbar("Y Pos", self.window_name, 300, 800, self._on_trackbar)
        cv2.createTrackbar("Width", self.window_name, 400, 800, self._on_trackbar)
        cv2.createTrackbar("Height", self.window_name, 600, 1000, self._on_trackbar)

        while True:
            display_frame = frame.copy()
            
            # 1. Read the user's intuitive 2D inputs from trackbars
            # Subtracting offsets to allow negative movement relative to the calibrated center
            x = cv2.getTrackbarPos("X Pos", self.window_name) - 200
            y = cv2.getTrackbarPos("Y Pos", self.window_name) - 300
            w = cv2.getTrackbarPos("Width", self.window_name)
            h = cv2.getTrackbarPos("Height", self.window_name)

            # 2. Define the corners of the ideal canonical rectangular strike zone
            canonical_pts = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype="float32").reshape(-1, 1, 2)

            # 3. WARP the canonical points to the camera's skewed perspective!
            # This perfectly projects our flat 2D strike zone box into the diagonal camera view!
            skewed_pts = cv2.perspectiveTransform(canonical_pts, self.H)
            skewed_pts = np.int32(skewed_pts)

            # Draw the skewed polygon on the camera feed
            cv2.polylines(display_frame, [skewed_pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Add instruction overlay
            cv2.putText(display_frame, "Adjust Zone using Trackbars. Press 'q' to save & exit.", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return (x, y, w, h)


if __name__ == "__main__":
    print("Testing Phase 2 Setup...")
    
    # URL for Camera 2 (Judge Camera)
    # Using the same IP address as in main.py
    cam2_url = "http://192.168.129.70:8080/video"
    print(f"Connecting to live camera feed at {cam2_url}...")
    
    cap = cv2.VideoCapture(cam2_url)
    if not cap.isOpened():
        print(f"Error: Could not open live camera feed. Ensure IP Webcam is running at {cam2_url}.")
        exit(1)
        
    print("Waiting for a clean live frame...")
    # Buffer a few frames to clear the stream and get the latest
    for _ in range(10):
        ret, frame = cap.read()
        
    cap.release()
    
    if not ret or frame is None:
        print("Error: Could not read a frame. Please check your network connection.")
        exit(1)
        
    # 2. Run Calibration using the LIVE frame!
    calibrator = StrikeZoneCalibrator(frame)
    H = calibrator.calibrate()
    
    if H is not None:
        # Save Homography for future sessions!
        np.save("homography.npy", H)
        print("-> Saved homography.npy directly to disk.")
        
        # 3. Run Adjustable GUI using the calibrated Homography and live frame
        gui = StrikeZoneGUI(H)
        try:
            final_zone = gui.run_gui(frame)
            print(f"-> Final Canonical Strike Zone parameters locked in: X={final_zone[0]} Y={final_zone[1]} W={final_zone[2]} H={final_zone[3]}")
        except cv2.error:
            # Handle if you close the window with 'X' instead of 'q'
            print("GUI window closed manually.")
    else:
        print("Calibration was cancelled.")
