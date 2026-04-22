import cv2
import time


class Draw:

    def draw_countdown(self, frame, game):
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
    def draw_target(self, frame, game):
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
    def draw_hit_effect(self, frame, game):
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
        h, w, _ = frame.shape

        cv2.putText(frame,f"Score: {game.score}",(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),3)

        bar_height = int(min(game.score * 8, 300))

        cv2.rectangle(frame,(w - 50, h - bar_height),(w - 20, h),(0, 255, 0),-1)

        cv2.putText(frame,f"COMBO x{game.combo}",(10, 130),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),3)

    # ================= MISS EFFECT =================
    def draw_miss_effects(self, frame, game):
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
    def draw_position_line(self, frame, game):
        h, w, _ = frame.shape

        line_y = int(game.target_y + 40)

        overlay = frame.copy()

        cv2.line(overlay, (100, line_y), (w - 100, line_y), (255, 255, 0), 6)
        cv2.line(overlay, (120, line_y + 5), (w - 120, line_y + 5), (200, 200, 0), 2)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame,
                    "Stand behind this line",(int(w * 0.25), line_y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0),2)

    def draw_boost_overlay(self, frame, game):
        if not game.boost_mode:
            return

        overlay = frame.copy()

        color = (0, 255, 0) if game.score_multiplier > 1 else (0, 0, 255)
        alpha = 0.12

        cv2.rectangle(overlay, (0, 0), (640, 480), color, -1)

        cv2.addWeighted(overlay, alpha, frame,1 - alpha, 0, frame)