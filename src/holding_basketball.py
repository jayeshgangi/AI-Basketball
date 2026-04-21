import cv2
from ultralytics import YOLO
import numpy as np
import time,logging,os


os.makedirs("logs",exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO in production
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ball_holding_debug.log", mode="w",encoding="utf-8")
    ]
)

logger = logging.getLogger("BallHoldingDetector")


class BallHoldingDetector:
    def __init__(self):
        
        logger.info("Initializing BallHoldingDetector...")

        try:
            self.pose_model = YOLO("models/yolov8s-pose.pt")
            self.ball_model = YOLO("models/basketballModel.pt")
            logger.info("Model loaded successfully.")
        
        except Exception as e:
            logger.exception("Failed to load models!")
            raise e

        # Open the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Webcam failed to open!")
            raise RuntimeError("Camera not accessible")
        logger.info("Webcam initialized.")

        # Define the body part indices. Switch left and right to account for the mirrored image.
        self.body_index = {
            "left_wrist": 10,  # switched
            "right_wrist": 9,  # switched
            }

        self.hold_start_time = None
        self.is_holding = False

        self.hold_duration = 0.85
        self.hold_threshold = 300

        # Debug toggles
        self.debug_overlay = True

        # Performance tracking
        self.prev_time = time.time()


    def run(self):

        logger.info("Starting main loop....")

        while self.cap.isOpened():

            loop_start = time.time()

            success, frame = self.cap.read()

            if not success:
                logger.warning("Frame capture failed.")
                break

            try:
                # Process the current frame
                pose_annotated_frame, ball_detected = self.process_frame(frame)
            except Exception as e:
                logger.exception("Error during frame processing!")
                continue

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            logger.debug(f"FPS: {fps:.2f} | Ball detected: {ball_detected}")

            cv2.putText(
                pose_annotated_frame,
                f"FPS: {fps:.2f}",(10, 130),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)

            cv2.imshow("YOLOv8 Inference", pose_annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exit key pressed.")
                break

            logger.debug(f"Frame processing time: {time.time() - loop_start:.4f}s")

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Application closed cleanly.")


    def process_frame(self, frame):

        logger.debug("Processing new frame...")

        # Perform pose estimation on the frame
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        keypoints = pose_results[0].keypoints

        if keypoints is None or keypoints.xy is None:
            logger.warning("No keypoints detected.")
            return pose_annotated_frame, False

        # Proper extraction
        rounded_results = keypoints.xy.cpu().numpy()

        if len(rounded_results) == 0:
            logger.warning("No human detected.")
            return pose_annotated_frame, False

        person = rounded_results[0]  # first detected person

        if len(person) <= max(self.body_index.values()):
            logger.warning("Incomplete keypoints.")
            return pose_annotated_frame, False

        left_wrist = person[self.body_index["left_wrist"]]
        right_wrist = person[self.body_index["right_wrist"]]

        logger.debug(f"Left wrist: {left_wrist}, Right wrist: {right_wrist}")

        # Perform ball detection on the frame
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.25)

        # Set the ball detection flag to False before the detection
        ball_detected = False

        for ball_results in ball_results_list:
            if ball_results.boxes is None:
                logger.debug("No bounding boxes in the result.")
                continue

            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                logger.debug(f"Ball center: ({ball_x_center:.2f}, {ball_y_center:.2f})")

                # Update the ball detection flag to True when the ball is detected
                ball_detected = True

                # Calculate distances between the ball and the wrists
                left_distance = np.hypot(ball_x_center - left_wrist[0], ball_y_center - left_wrist[1])
                
                right_distance = np.hypot(ball_x_center - right_wrist[0], ball_y_center - right_wrist[1])

                logger.debug(f"Distances -> Left: {left_distance:.2f}, Right: {right_distance:.2f}")

                # Check if the ball is being held
                self.check_holding(left_distance, right_distance)

                # Annotate ball detection on the pose estimation frame
                cv2.rectangle(
                    pose_annotated_frame,(int(x1), int(y1)),(int(x2), int(y2)),(0, 255, 0),2,)

                if self.debug_overlay:
                    self.draw_debug_info(pose_annotated_frame,ball_x_center,ball_y_center,left_wrist,right_wrist,left_distance,right_distance)
                
                # Blue tint if holding
                if self.is_holding:
                    logger.info("Holding detected -> applying blue overlay.")
                    blue_tint = np.full_like(pose_annotated_frame, (255, 0, 0), dtype=np.uint8)
                    pose_annotated_frame = cv2.addWeighted(pose_annotated_frame, 0.7, blue_tint, 0.3, 0)
                
        # If the ball is not detected in the frame, reset the timer and the holding flag
        if not ball_detected:
            logger.debug("No ball detected -> resetting state.")
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected
    

    def draw_debug_info(self,frame,ball_x,ball_y,left_wrist,right_wrist,left_distance,right_distance
    ):
        cv2.putText(frame, f"Ball: ({ball_x:.1f},{ball_y:.1f})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"LW: ({left_wrist[0]:.1f},{left_wrist[1]:.1f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"RW: ({right_wrist[0]:.1f},{right_wrist[1]:.1f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"MinDist: {min(left_distance, right_distance):.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Holding: {self.is_holding}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)



    def check_holding(self, left_distance, right_distance):

        min_dist = min(left_distance, right_distance)

        logger.debug(f"Checking holding condition: min_dist={min_dist:.2f}")

        if min_dist < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                logger.debug("Hold timer started.")

            elif time.time() - self.hold_start_time > self.hold_duration:
                if not self.is_holding:
                    logger.info("BALL IS BEING HELD ")
                self.is_holding = True
        else:
            if self.is_holding:
                logger.info("Ball released.")
            self.hold_start_time = None
            self.is_holding = False


if __name__ == "__main__":
    ball_detection = BallHoldingDetector()
    ball_detection.run()
