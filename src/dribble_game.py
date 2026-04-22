import cv2
import numpy as np
import time
import random
import logging
from ultralytics import YOLO
from src.dribble_counting import DribbleCounter
from utils.Draw import Draw
from typing import Optional,Tuple,List,Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/game.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()])

logger = logging.getLogger("DribbleGame")

class DribbleGame:
    def __init__(self) -> None:
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

        self.game_duration = 60  # seconds
        self.match_start_time = None
        self.game_over = False
        self.final_screen = False
        self.max_combo=0

        self.generate_new_target()

    def generate_new_target(self) -> None:
        
        """
        Generate a new target position on the screen.

        The target is placed randomly along the horizontal axis,
        near the bottom of the frame. Also resets the zone timer.

        Returns:
            None
        """

        w, h = 640, 480

        self.target_x = random.randint(100, w - 100)
        self.target_y = int(h * 0.90)

        self.zone_start_time = time.time()

        logger.info(f"New target at ({self.target_x}, {self.target_y})")

    def is_inside_target(self, x : float, y : float ,ball_radius : float) -> bool: 

        """
        Check if the ball bounce point is inside the target zone.

        Args:
            x (float): X-coordinate of the bounce point.
            y (float): Y-coordinate of the bounce point.
            ball_radius (float): Radius of the detected ball.

        Returns:
            bool: True if inside the target zone, False otherwise.
        """

        visual_y = self.target_y + 15 
        dist = np.hypot(x - self.target_x, y - visual_y)

        return dist < (self.target_radius + ball_radius * 0.6)
    
    def update_boost_mode(self) -> None:

        """
        Update boost mode state and scoring multiplier.

        Handles:
        - Boost duration expiration
        - Score multiplier update
        - Combo reset if inactivity threshold is reached

        Returns:
            None
        """
        
        if self.boost_mode:
            if time.time() - self.boost_start_time > self.boost_duration:
                self.boost_mode = False
        
        self.score_multiplier = 3 if self.boost_mode else 1

        if time.time() - self.last_hit_time > 2.5:
            self.combo = 0

    def print_final_stats(self) -> None:

        """
        Print final game statistics to the console.

        Displays:
        - Final score
        - Maximum combo achieved

        Returns:
            None
        """
        print("\n======================")
        print(" GAME ENDED")
        print(f" FINAL SCORE: {self.score}")
        print(f" MAX COMBO: {self.max_combo}")
        print("======================\n")

    def run(self) -> None:
        """
            Run the main game loop.

            Handles:
            - Camera frame capture
            - Countdown and game start
            - Object detection using YOLO
            - Dribble detection and scoring
            - Boost mode logic
            - Game timer and game over condition
            - Rendering UI elements

            Loop exits when:
            - User presses 'Q'
            - Camera feed ends
            - Keyboard interrupt occurs

            Returns:
                None
        """
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))

                now = time.time()

                # ----------------------------
                # START TIMER (ONLY ONCE)
                # ----------------------------
                if self.game_started and self.match_start_time is None:
                    self.match_start_time = now

                # ----------------------------
                # GAME TIMER CHECK (ALWAYS RUNS)
                # ----------------------------
                if self.match_start_time is not None and not self.game_over:
                    if now - self.match_start_time >= self.game_duration:
                        self.game_over = True
                        logger.info("GAME OVER TRIGGERED")

                # ----------------------------
                # GAME OVER SCREEN (FREEZE UI)
                # ----------------------------
                if self.game_over:
                    
                    frame = self.drawer.game_over_screen(frame,self)

                    cv2.imshow("Dribble game",frame)

                    key = cv2.waitKey(0)  # FREEZE SCREEN (IMPORTANT FIX)

                    if key == ord("q"):
                        break

                    continue

                # ----------------------------
                # BOOST LOGIC
                # ----------------------------
                if self.game_started:
                    if (not self.boost_mode and
                        now - self.last_boost_time > self.boost_cooldown and
                        random.random() < 0.002):

                        self.boost_mode = True
                        self.boost_start_time = now
                        self.score_multiplier = 3
                        self.last_boost_time = now

                        logger.info("BOOST ACTIVATED!")

                # ----------------------------
                # COUNTDOWN PHASE
                # ----------------------------
                if not self.game_started:
                    self.drawer.draw_position_line(frame, self)
                    self.game_started = self.drawer.draw_countdown(frame, self)

                    cv2.imshow("Dribble Game", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    continue

                # ----------------------------
                # MAIN GAMEPLAY
                # ----------------------------
                if self.game_started and not self.game_over:

                    # move target
                    if now - self.zone_start_time > self.zone_duration:
                        self.generate_new_target()

                    results_list = self.model.predict(frame, conf=0.25, verbose=False)
                    if not results_list:
                        continue

                    results = results_list[0]

                    if results.boxes is None or len(results.boxes.xyxy) == 0:
                        cv2.imshow("Dribble Game", frame)
                        continue

                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4].tolist()

                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        ball_radius = max((x2 - x1), (y2 - y1)) / 2

                        dribble, bounce_point = self.dribble_counter.update_dribble_count(
                            x_center, y_center
                        )

                        if dribble and bounce_point is not None:
                            bx, by = bounce_point

                            if self.is_inside_target(bx, by, ball_radius):

                                # combo logic
                                if now - self.last_hit_time < 1.2:
                                    self.combo = min(self.combo + 1, 10)
                                else:
                                    self.combo = 1
                                
                                self.max_combo = max(self.max_combo,self.combo)
                                self.last_hit_time = now

                                multiplier = self.score_multiplier * (1 + min(self.combo // 3, 3))

                                self.score += multiplier

                                self.hit_effect_time = now
                                self.hit_x = int(bx)
                                self.hit_y = int(by)

                                logger.info(f"HIT! Score: {self.score} Combo: {self.combo}")

                            else:
                                self.miss_effects.append({
                                    "x": int(bx),
                                    "y": int(by),
                                    "time": now
                                })

                                if len(self.miss_effects) > 30:
                                    self.miss_effects.pop(0)

                # ----------------------------
                # BOOST UPDATE
                # ----------------------------
                self.update_boost_mode()

                # ----------------------------
                # DRAW UI
                # ----------------------------
                self.drawer.draw_target(frame, self)
                self.drawer.draw_hit_effect(frame, self)
                self.drawer.draw_miss_effects(frame, self)
                self.drawer.draw_boost_overlay(frame, self)
                self.drawer.draw_ui(frame, self)

                cv2.imshow("Dribble Game", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ----------------------------
            # CLEAN EXIT
            # ----------------------------
            self.cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()

        except KeyboardInterrupt:
            self.cap.release()
            cv2.destroyAllWindows()

            print("\n======================")
            print(" GAME INTERRUPTED")
            print(f" FINAL SCORE: {self.score}")
            print("======================\n")

            logger.warning("Interrupted by user.")

if __name__ == "__main__":
    game = DribbleGame()
    game.run()