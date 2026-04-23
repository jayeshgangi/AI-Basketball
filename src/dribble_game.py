import numpy as np
import time,random,cv2,logging,os
from ultralytics import YOLO
from src.dribble_counting import DribbleCounter
from utils.Draw import Draw
from typing import Optional,Tuple,List,Dict
from dotenv import load_dotenv

load_dotenv()

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
        self.hit_effect_time = 0
        self.miss_effects=[]
        self.level = 1
        self.level2_threshold = 50
        self.level3_threshold = 120

        # Danger ball
        self.enemy_x = 100
        self.enemy_y = 100
        self.enemy_radius = 20
        self.enemy_hit_effect_time = 0
        self.enemy_dx = 5
        self.enemy_dy = 4

        # Target
        self.target_x = None
        self.target_y = None
        self.hit_x = None
        self.hit_y = None
        self.last_enemy_hit_time = 0
        self.enemy_hit_cooldown = 0.8   # seconds
        self.zone_start_time = time.time()
        self.game_start_time = time.time()
        self.game_started = False
        self.boost_mode = False
        self.boost_start_time = 0
        self.score_multiplier = 1
        self.combo = 0
        self.last_hit_time = 0
        self.last_boost_time = 0
        self.boost_cooldown = int(os.getenv("BOOST_COOLDOWN", 12))
        self.boost_duration = int(os.getenv("BOOST_DURATION", 7))
        self.countdown_duration = int(os.getenv("COUNTDOWN_DURATION", 5))
        self.zone_duration = int(os.getenv("ZONE_DURATION", 4))
        self.target_y_offset = float(os.getenv("TARGET_Y_OFFSET", 0.90))
        self.target_radius = int(os.getenv("TARGET_RADIUS", 35))
        self.game_duration = int(os.getenv("GAME_DURATION",60))
        self.boost_chance = float(os.getenv("BOOST_CHANCE", 0.002))
        self.combo_timeout = float(os.getenv("COMBO_TIMEOUT", 1.2))
        self.combo_decay_time = float(os.getenv("COMBO_DECAY_TIME", 1.5))
        self.max_combo_limit = int(os.getenv("MAX_COMBO_LIMIT", 10))
        self.combo_break_threshold = int(os.getenv("COMBO_BREAK_THRESHOLD", 3))
        self.yolo_conf = float(os.getenv("YOLO_CONFIDENCE", 0.25))
        self.score_step = int(os.getenv("SCORE_MULTIPLIER_STEP", 3))
        self.max_combo_bonus = int(os.getenv("MAX_COMBO_BONUS", 3))
        self.boost_multiplier = int(os.getenv("BOOST_MULTIPLIER", 3))
        self.match_start_time = None
        self.game_over = False
        self.max_combo=0
        self.last_decay_time = 0
        self.combo_break_time = 0
        self.enemy_spawn_time = 0

        self.generate_new_target()

    def update_level(self):
        prev_level = self.level

        if self.score >= self.level3_threshold:
            self.level = 3
        elif self.score >= self.level2_threshold:
            self.level = 2
        else:
            self.level = 1

        # 🔥 NEW: respawn enemy when entering level 2
        if prev_level < 2 and self.level >= 2:

            side = random.choice(["left", "right"])

            if side == "left":
                self.enemy_x = random.randint(40, 120)
            else:
                self.enemy_x = random.randint(520, 600)

            self.enemy_y = random.randint(40, 120)

            # 🔥 ensure it moves toward center
            self.enemy_dx = random.choice([4, 5])
            if side == "right":
                self.enemy_dx *= -1

            self.enemy_dy = random.choice([3, 4])
            self.enemy_spawn_time = time.time()

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
        self.target_y = int(h * self.target_y_offset)

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
    
    def update_enemy(self):
        if self.level < 2:
            return

        speed_multiplier = 1.5 + (self.level -1) * 1.0

        self.enemy_x += self.enemy_dx * speed_multiplier
        self.enemy_y += self.enemy_dy * speed_multiplier

        # Bounce off walls
        # Bounce off walls (FIXED)
        if self.enemy_x < self.enemy_radius or self.enemy_x > 640 - self.enemy_radius:
            self.enemy_dx *= -1

        if self.enemy_y < self.enemy_radius or self.enemy_y > 480 - self.enemy_radius:
            self.enemy_dy *= -1
    
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
        
        self.score_multiplier = self.boost_multiplier if self.boost_mode else 1 

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
                        random.random() < self.boost_chance):

                        self.boost_mode = True
                        self.boost_start_time = now
                        self.score_multiplier = self.boost_multiplier
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

                    self.update_level()

                    # move target
                    if now - self.zone_start_time > self.zone_duration:
                        self.generate_new_target()

                    results_list = self.model.predict(frame, conf=self.yolo_conf, verbose=False)
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

                        dribble, bounce_point = self.dribble_counter.update_dribble_count(x_center, y_center)

                        check_x, check_y = x_center, y_center
                        if bounce_point is not None:
                            check_x = 0.6 * bounce_point[0] + 0.4 * x_center
                            check_y = 0.6 * bounce_point[1] + 0.4 * y_center

                        ignore_collision = (time.time() - self.enemy_spawn_time < 0.5)

                        if self.level >= 2 and not ignore_collision:

                            dist_enemy = np.hypot(check_x - self.enemy_x,check_y - self.enemy_y)

                            collision_threshold = self.enemy_radius + ball_radius + 20

                            if dist_enemy < collision_threshold:
                                if now - self.last_enemy_hit_time > self.enemy_hit_cooldown:

                                    self.score = max(0, self.score - 5)
                                    self.score = min(self.score, self.level2_threshold - 1) 

                                    self.level = 1
                                    self.combo = 0
                                    self.last_hit_time = 0

                                    self.enemy_hit_effect_time = now

                                    # push enemy away so no repeated instant hits
                                    self.enemy_x = -100
                                    self.enemy_y = -100

                                    self.last_enemy_hit_time = now

                                    logger.info("HIT BY ENEMY! RESET TO LEVEL 1")

                                    continue

                        if dribble and bounce_point is not None:
                            bx, by = bounce_point

                            if self.is_inside_target(bx, by, ball_radius):

                                if now - self.last_hit_time < self.combo_timeout:
                                    self.combo = min(self.combo + 1, self.max_combo_limit)
                                else:
                                    # COMBO BREAK DETECTED
                                    if self.combo >= self.combo_break_threshold:
                                        self.combo_break_time = now

                                    self.combo = 1
                                
                                self.last_hit_time = now

                                self.max_combo = max(self.max_combo,self.combo)

                                multiplier = self.score_multiplier * (1 + min(self.combo // self.score_step, self.max_combo_bonus))

                                self.score += multiplier

                                self.hit_effect_time = now
                                self.hit_x = int(bx)
                                self.hit_y = int(by)

                                logger.info(f"HIT! Score: {self.score} Combo: {self.combo}")
                                print(f"Combo: {self.combo}, Time since last hit: {now - self.last_hit_time:.2f}")

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

                # COMBO DECAY (SMOOTH)
                # -------------------------
                if self.last_hit_time != 0 and now - self.last_hit_time > self.combo_decay_time:
                    if now - self.last_decay_time > 0.2:
                        if self.combo > 0:
                            self.combo -= 1
                            self.last_decay_time = now

                # ----------------------------
                # Move enemy (only works after level update)
                self.update_enemy()
                # DRAW UI
                # ----------------------------
                self.drawer.draw_target(frame, self)
                self.drawer.draw_enemy(frame,self)
                self.drawer.draw_hit_effect(frame, self)
                self.drawer.draw_enemy_hit_flash(frame, self)
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