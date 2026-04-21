import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging, os


os.makedirs("logs",exist_ok=True)

# =========================
# LOGGER CONFIGURATION
# =========================
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO for less noise
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/dribble_debug.log", mode="w",encoding="utf-8")
    ]
)

logger = logging.getLogger("DribbleCounter")


class DribbleCounter:
    def __init__(self):
        logger.info("Initializing DribbleCounter...")

        # Load model
        try:
            self.model = YOLO("models/basketballModel.pt")
            logger.info("Ball detection model loaded.")
        except Exception as e:
            logger.exception("Failed to load YOLO model!")
            raise e

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Webcam failed to open!")
            raise RuntimeError("Camera not accessible")
        logger.info("Webcam initialized.")

        # Tracking state
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        
        #  ADDEDD
        self.lowest_point = None
        #   .......

        # Counter
        self.dribble_count = 0

        # Threshold
        self.dribble_threshold = 18

        # Debug toggles
        self.debug_overlay = True

        # Performance
        self.prev_time = time.time()

    def run(self):
        logger.info("Starting dribble detection loop...")

        try:
            while self.cap.isOpened():
                loop_start = time.time()

                success, frame = self.cap.read()

                if not success:
                    logger.warning("Frame capture failed.")
                    break

                try:
                    annotated_frame, detected = self.process_frame(frame)
                except Exception as e:
                    logger.exception("Error during frame processing!")
                    continue

                # FPS
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                logger.debug(f"FPS: {fps:.2f} | Ball detected: {detected}")

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.2f}",(10, 80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)

                cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exit key pressed.")
                    break

                logger.debug(f"Frame time: {time.time() - loop_start:.4f}s")

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")

        finally:
            logger.info(f"Total Dribbles detected: {self.dribble_count}")
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Application closed cleanly.")

    def process_frame(self, frame):
        logger.debug("Processing frame...")

        results_list = self.model(frame, verbose=False, conf=0.25)

        ball_detected = False
        annotated_frame = frame.copy()

        for results in results_list:
            if results.boxes is None:
                logger.debug("No bounding boxes in results.")
                continue

            annotated_frame = results.plot()

            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                logger.debug(f"Ball center: ({x_center:.2f}, {y_center:.2f})")

                ball_detected = True

                self.update_dribble_count(x_center, y_center)

                self.prev_x_center = x_center
                self.prev_y_center = y_center

                if self.debug_overlay:
                    self.draw_debug_info(annotated_frame,x_center,y_center)

        if not ball_detected:
            logger.debug("No ball detected -> resetting motion state.")
            self.prev_delta_y = None

        # UI
        cv2.putText(
            annotated_frame,
            f"Dribbles: {self.dribble_count}",(10, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)

        return annotated_frame, ball_detected

    def draw_debug_info(self, frame, x_center, y_center):
        cv2.putText(frame,f"Ball: ({x_center:.1f},{y_center:.1f})",(10, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)

        if self.prev_delta_y is not None:
            cv2.putText(
                frame,
                f"deltaY: {self.prev_delta_y:.2f}",(10, 35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)

    # def update_dribble_count(self, x_center, y_center):
    #     logger.debug("Updating dribble count...")

    #     if self.prev_y_center is not None:
    #         delta_y = y_center - self.prev_y_center

    #         logger.debug(
    #             f"delta_y: {delta_y:.2f}, prev_delta_y: {self.prev_delta_y}"
    #         )

    #         if self.prev_delta_y is not None:

    #             # DOWN → UP transition detection
    #             if self.prev_delta_y > 5 and delta_y < -5:
    #                 self.dribble_count += 1
    #                 logger.info(
    #                     f"DRIBBLE DETECTED | Count: {self.dribble_count}"
    #                 )

    #             else:
    #                 logger.debug("No valid direction change.")

    #         self.prev_delta_y = delta_y

    #     else:
    #         logger.debug("First frame — initializing position.")

    #     self.prev_y_center = y_center

    def update_dribble_count(self, x_center, y_center):
        dribble_detected = False
        bounce_point = None

        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            # -------------------------
            # TRACK LOWEST POINT (falling phase)
            # -------------------------
            if delta_y > 0:  # ball going DOWN
                if self.lowest_point is None or y_center > self.lowest_point[1]:
                    self.lowest_point = (x_center, y_center)

            # -------------------------
            # DETECT DRIBBLE (DOWN → UP)
            # -------------------------
            if self.prev_delta_y is not None:
                if self.prev_delta_y > 5 and delta_y < -5:
                    dribble_detected = True
                    self.dribble_count += 1

                    # Use lowest point as bounce point
                    bounce_point = self.lowest_point

                    # RESET for next dribble
                    self.lowest_point = None

            # Store delta
            self.prev_delta_y = delta_y

        # Update previous position
        self.prev_y_center = y_center

        return dribble_detected, bounce_point
    
if __name__ == "__main__":
    counter = DribbleCounter()
    counter.run()