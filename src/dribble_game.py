import cv2
import numpy as np
import time
import random
import logging
from ultralytics import YOLO
from src.dribble_counting import DribbleCounter
from utils.Draw import Draw


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/game.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DribbleGame")


class DribbleGame:
    def __init__(self):
        logger.info("Starting Game...")

        self.model = YOLO("models/basketballModel.pt")
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.dribble_counter = DribbleCounter()
        self.drawer = Draw()

        # Game state
        self.score = 0
        self.score_history = []
        self.hit_effect_time = 0
        self.miss_effects=[]

        # Target
        self.target_x = None
        self.target_y = None
        self.target_radius = 35
        self.hit_x = None
        self.hit_y = None

        self.zone_start_time = time.time()
        self.zone_duration = 4

        self.game_start_time = time.time()
        self.countdown_duration = 5  # seconds
        self.game_started = False


        self.boost_mode = False
        self.boost_start_time = 0
        self.boost_duration = 7  # 5–10 sec
        self.score_multiplier = 1
        self.combo = 0
        self.last_hit_time = 0
        self.last_boost_time = 0
        self.boost_cooldown = 12

        self.generate_new_target()

    def generate_new_target(self):
        w, h = 640, 480

        self.target_x = random.randint(100, w - 100)
        self.target_y = int(h * 0.90)

        self.zone_start_time = time.time()

        logger.info(f"New target at ({self.target_x}, {self.target_y})")

    def is_inside_target(self, x, y,ball_radius):
        visual_y = self.target_y + 15 
        dist = np.hypot(x - self.target_x, y - visual_y)

        return dist < (self.target_radius + ball_radius * 0.6)
    
    # def get_speed(self):
    #     now = time.time()
    #     self.score_history = [t for t in self.score_history if now - t < 3]
    #     return len(self.score_history) / 3.0
    
    def update_boost_mode(self):
        
        if self.boost_mode:
            if time.time() - self.boost_start_time > self.boost_duration:
                self.boost_mode = False
        
        self.score_multiplier = 3 if self.boost_mode else 1

        if time.time() - self.last_hit_time > 2.5:
            self.combo = 0

    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()

                if not success:
                    break

                frame = cv2.flip(frame,1)

                frame = cv2.resize(frame, (640, 480))

                if self.game_started:
                    now = time.time()
                    if (not self.boost_mode and now - self.last_boost_time > self.boost_cooldown and random.random() < 0.002):

                        self.boost_mode = True
                        self.boost_start_time = time.time()
                        self.score_multiplier = 3
                        self.last_boost_time = now

                        logger.info("BOOST ACTIVATED!")

                if not self.game_started:
                    self.drawer.draw_position_line(frame,self)
                    self.game_started = self.drawer.draw_countdown(frame,self)

                    cv2.imshow("Dribble Game", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    continue

                # Move target
                if time.time() - self.zone_start_time > self.zone_duration:
                    self.generate_new_target()

                results_list = self.model.predict(frame, conf=0.25, verbose=False)

                if len(results_list)==0:
                    continue

                results = results_list[0]

                if results.boxes is None or len(results.boxes) == 0:
                    print("No detections")
                    continue

                for bbox in results.boxes.xyxy:
                    x1, y1, x2, y2 = bbox[:4].tolist()

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    ball_radius = max((x2 - x1),(y2 - y1)) /2

                    dribble,bounce_point = self.dribble_counter.update_dribble_count(
                        x_center, y_center
                    )
                    if dribble and bounce_point is not None:
                        bx, by = bounce_point

                        if self.is_inside_target(bx, by,ball_radius):

                            now = time.time()

                            # combo logic
                            if now - self.last_hit_time < 1.2:
                                self.combo +=1
                                if self.combo > 10:
                                    self.combo = 10
                            else:
                                self.combo = 1

                            self.last_hit_time = now

                            multiplier = self.score_multiplier * (1 + min(self.combo // 3,3))

                            self.score += multiplier
                            self.score_history.append(time.time())

                            self.hit_effect_time = time.time()
                            self.hit_x = int(bx)
                            self.hit_y = int(by)
                            logger.info(f"HIT! Score: {self.score} Combo: {self.combo}")
                        
                        else:
                            self.miss_effects.append({
                                "x": int(bx),
                                "y": int(by),
                                "time": time.time()
                            })
                            if len(self.miss_effects) > 30:
                                self.miss_effects.pop(0)

                self.update_boost_mode()

                # Draw UI
                self.drawer.draw_target(frame,self)
                self.drawer.draw_hit_effect(frame,self)
                self.drawer.draw_miss_effects(frame,self)
                self.drawer.draw_boost_overlay(frame,self)
                self.drawer.draw_ui(frame,self)

                cv2.imshow("Dribble Game", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")


if __name__ == "__main__":
    game = DribbleGame()
    game.run()