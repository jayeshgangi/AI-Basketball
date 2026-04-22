from __future__ import annotations
import cv2,time
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dribble_game import DribbleGame

class Draw:

    def draw_countdown(self, frame : np.ndarray, game : DribbleGame) -> bool:
        """
        Render countdown before the game starts.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing timing state.

        Returns:
            bool: True when countdown finishes and game should start,
                False otherwise.
        """
        elapsed = time.time() - game.game_start_time
        remaining = int(game.countdown_duration - elapsed) + 1

        if remaining > 0:
            h, w, _ = frame.shape

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            text = str(remaining)

            cv2.putText(frame, text,(w // 2 - 50, h // 2),cv2.FONT_HERSHEY_SIMPLEX,6,(0, 255, 255),10)

            return False

        return True

    # ================= TARGET =================
    def draw_target(self, frame : np.ndarray, game:DribbleGame) -> None:
        """
        Draw the current target zone on screen.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing target position.

        Returns:
            None
        """
        x, y = int(game.target_x), int(game.target_y + 15)

        axes = (int(game.target_radius * 1.6),
                int(game.target_radius * 0.5))

        cv2.ellipse(frame, (x, y + 3), axes, 0, 0, 360, (0, 0, 80), -1)

        segments = 24
        for i in range(segments):
            if i % 2 == 0:
                start = int((i / segments) * 360)
                end = int(((i + 1) / segments) * 360)

                cv2.ellipse(frame, (x, y), axes, 0, start, end, (0, 0, 255), 3)

        cv2.ellipse(frame,(x, y),(int(axes[0] * 0.65), int(axes[1] * 0.65)),0, 0, 360,(0, 0, 180),2)

    # ================= HIT EFFECT =================
    def draw_hit_effect(self, frame: np.ndarray, game: DribbleGame) -> None:
        """
        Draw visual feedback when a target is hit.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing hit effect data.

        Returns:
            None
        """
        if time.time() - game.hit_effect_time < 0.15 and game.hit_x is not None:
            overlay = frame.copy()

            progress = (time.time() - game.hit_effect_time) / 0.15
            progress = min(progress, 1.0)

            scale = 1 + 0.3 * (1 - progress)

            radius_x = int((game.target_radius + 10) * scale)
            radius_y = int(radius_x * 0.5)

            cv2.ellipse(overlay,(game.hit_x, game.hit_y),(radius_x, radius_y),0, 0, 360,(0, 255, 0),-1)

            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # ================= UI =================
    def draw_ui(self, frame, game):

        """
        Render main game UI elements.

        Includes:
        - Score
        - Combo counter
        - Timer (if implemented)

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance with UI data.

        Returns:
            None
        """

        if game.game_started and not game.game_over:

            if game.match_start_time is not None:
                elapsed = time.time() - game.match_start_time
                remaining = max(0, int(game.game_duration - elapsed))
            else:
                remaining = game.game_duration

            cv2.putText(frame, f"Time: {remaining}",(500, 40),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 3)

            h, w, _ = frame.shape

            cv2.putText(frame, f"Score: {game.score}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 3)

            bar_height = min(int(game.score * 8), h - 60)

            cv2.rectangle(frame,(w - 50, h - bar_height),(w - 20, h),(0, 255, 0), -1)

            cv2.putText(frame, f"COMBO x{game.combo}",(10, 90),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 3)
            
    # ================= MISS EFFECT =================
    def draw_miss_effects(self, frame : np.ndarray, game : DribbleGame) -> None:
        """
        Draw visual indicators for missed shots.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing miss effects list.

        Returns:
            None
        """
        
        new_effects = []

        for effect in game.miss_effects:
            age = time.time() - effect["time"]

            if age < 0.5:
                progress = age / 0.5
                radius = int(10 + 50 * progress)
                alpha = 1 - progress

                overlay = frame.copy()

                cv2.circle(overlay,(effect["x"], effect["y"] + 10),radius,(0, 0, 255),3)

                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                new_effects.append(effect)

        game.miss_effects = new_effects

    # ================= POSITION LINE =================
    def draw_position_line(self, frame : np.ndarray, game : DribbleGame) -> None:

        """
        Draw a guide line to help player positioning.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance (for dimensions if needed).

        Returns:
            None
        """
        h, w, _ = frame.shape

        line_y = int(game.target_y + 40)

        overlay = frame.copy()

        cv2.line(overlay, (100, line_y), (w - 100, line_y), (255, 255, 0), 6)
        cv2.line(overlay, (120, line_y + 5), (w - 120, line_y + 5), (200, 200, 0), 2)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame,
                    "Stand behind this line",(int(w * 0.25), line_y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0),2)

    def draw_boost_overlay(self, frame : np.ndarray, game : DribbleGame) -> None:
        """
        Display boost mode visual overlay when active.

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing boost state.

        Returns:
            None
        """
        if not game.boost_mode:
            return

        overlay = frame.copy()

        color = (0, 255, 0) if game.score_multiplier > 1 else (0, 0, 255)
        alpha = 0.12

        cv2.rectangle(overlay, (0, 0), (640, 480), color, -1)

        cv2.addWeighted(overlay, alpha, frame,1 - alpha, 0, frame)

    def game_over_screen(self, frame : np.ndarray, game : DribbleGame) -> np.ndarray:

        """
        Render the game over screen.

        Displays:
        - "TIME'S UP" message
        - Final score
        - Exit instruction

        Args:
            frame (np.ndarray): Current video frame.
            game (DribbleGame): Game instance containing final score.

        Returns:
            np.ndarray: Frame with game over overlay.
        """

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 1.0, frame, 0, 0)

        cv2.putText(frame, "TIME'S UP!", (140, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        cv2.putText(frame, "Well Played!", (160, 260),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(frame, f"Final Score: {game.score}", (150, 330),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        cv2.putText(frame, "Press Q to exit", (170, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        return frame
