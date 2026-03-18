import cv2
import threading
import time

class DualCameraStream:
    """
    Handles two video streams simultaneously using threading.
    Ensures that capturing frames from Camera 1 doesn't block Camera 2.
    Pairs frames based on the closest timestamp to minimize sync issues.
    """
    def __init__(self, src1=0, src2=1):
        # Open video streams (0 and 1 are default camera indices, can be video file paths)
        self.stream1 = cv2.VideoCapture(src1)
        self.stream2 = cv2.VideoCapture(src2)
        
        # Check if cameras/videos opened successfully
        if not self.stream1.isOpened():
            print(f"Warning: Unable to open Camera 1 at source: {src1}")
        if not self.stream2.isOpened():
            print(f"Warning: Unable to open Camera 2 at source: {src2}")
            
        self.frame1 = None
        self.frame2 = None
        self.timestamp1 = 0
        self.timestamp2 = 0
        
        self.stopped = False
        
        # Locks to prevent reading partially updated frames
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()

    def start(self):
        """Starts the threads for reading frames from both cameras continuously."""
        self.thread1 = threading.Thread(target=self._update_cam1, daemon=True)
        self.thread2 = threading.Thread(target=self._update_cam2, daemon=True)
        self.thread1.start()
        self.thread2.start()
        return self

    def _update_cam1(self):
        """Continuously reads frames from Camera 1 in a background thread."""
        while not self.stopped:
            grabbed, frame = self.stream1.read()
            if grabbed:
                with self.lock1:
                    self.frame1 = frame
                    self.timestamp1 = time.perf_counter()
            else:
                self.stopped = True

    def _update_cam2(self):
        """Continuously reads frames from Camera 2 in a background thread."""
        while not self.stopped:
            grabbed, frame = self.stream2.read()
            if grabbed:
                with self.lock2:
                    self.frame2 = frame
                    self.timestamp2 = time.perf_counter()
            else:
                self.stopped = True

    def get_paired_frames(self):
        """
        Returns the most recently captured frames from both cameras and their timestamps.
        Pairing relies on grabbing the latest available frames which are continuously updated.
        """
        # Safely extract the latest frames and their timestamps
        with self.lock1:
            f1, t1 = self.frame1, self.timestamp1
        with self.lock2:
            f2, t2 = self.frame2, self.timestamp2
            
        # Copy frames to avoid issues if the original NumPy arrays get modified later
        if f1 is not None and f2 is not None:
             return f1.copy(), f2.copy(), t1, t2
        return None, None, t1, t2

    def stop(self):
        """Stops the camera streams and joins threads to safely exit."""
        self.stopped = True
        if hasattr(self, 'thread1') and self.thread1.is_alive():
            self.thread1.join()
        if hasattr(self, 'thread2') and self.thread2.is_alive():
            self.thread2.join()
        if self.stream1.isOpened():
            self.stream1.release()
        if self.stream2.isOpened():
            self.stream2.release()


if __name__ == "__main__":
    # Example Usage
    print("Initializing Dual Camera Stream...")
    # NOTE: You can replace 0 and 1 with actual video file paths for offline testing
    # E.g., src1="camera1_side.mp4", src2="camera2_diagonal.mp4"
    cam_stream = DualCameraStream(src1=0, src2=1)
    cam_stream.start()
    
    # Allow cameras to warm up
    time.sleep(2.0)
    
    try:
        while True:
            # Grab the latest, most synchronized paired frames
            frame1, frame2, t1, t2 = cam_stream.get_paired_frames()
            
            if frame1 is not None and frame2 is not None:
                # Calculate absolute time difference between the captured frames
                time_diff = abs(t1 - t2)
                
                # Display the frames
                cv2.imshow("Camera 1 (Side View - Trigger)", frame1)
                cv2.imshow("Camera 2 (Diagonal Behind View - Judge)", frame2)
                
                # Print sync accuracy to terminal occasionally to monitor latency
                # print(f"Inter-frame sync latency: {time_diff * 1000:.2f} ms")
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    finally:
        cam_stream.stop()
        cv2.destroyAllWindows()
