import os,cv2,time

class Draw():

    def draw_countdown(self, frame,game):
        elapsed = time.time() - game.game_start_time
        remaining = int(game.countdown_duration - elapsed) + 1

        if remaining > 0:
            h, w, _ = frame.shape

            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Big number
            text = str(remaining)
            font_scale = 6
            thickness = 10

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, thickness)[0]

            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2

            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 255),
                        thickness)

            return False  # game not started yet

        return True  # start game

    def draw_target(self, frame,game):
        x, y = int(game.target_x), int(game.target_y + 15)

        axes = (int(game.target_radius * 1.6), int(game.target_radius * 0.5))

        cv2.ellipse(frame,(x, y + 3),axes,0,0,360,(0, 0, 80),-1)

        # --- DASHED OUTER RING (cleaner spacing) ---
        segments = 24
        for i in range(segments):
            if i % 2 == 0:
                start = int((i / segments) * 360)
                end = int(((i + 1) / segments) * 360)

                cv2.ellipse(frame,(x, y),axes,0,start,end,(0, 0, 255),3)

        # --- INNER FADE RING ---
        cv2.ellipse(frame,(x, y),(int(axes[0] * 0.65), int(axes[1] * 0.65)),0,0,360,(0, 0, 180),2)

    def draw_hit_effect(self, frame,game):
        if time.time() - game.hit_effect_time < 0.15 and game.hit_x is not None:
            overlay = frame.copy()

            progress = (time.time() - game.hit_effect_time) / 0.15
            progress = min(progress, 1.0)

            scale = 1 + 0.3 * (1 - progress)

            radius_x = int((game.target_radius + 10) * scale)
            radius_y = int(radius_x * 0.5)

            cv2.ellipse(overlay,(game.hit_x, game.hit_y),(radius_x, radius_y),0,0,360,(0, 255, 0),-1)

            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


    def draw_ui(self, frame,game):
            h, w, _ = frame.shape

            # Score
            cv2.putText(frame, f"Score: {game.score}", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Speed bar
            speed = game.get_speed()
            bar_height = int(min(speed * 120, 300))

            cv2.rectangle(frame, (w - 50, h - bar_height),(w - 20, h), (0, 255, 0), -1)

            cv2.putText(frame, f"{speed:.2f} d/s",(w - 150, h - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)

    def draw_miss_effects(self, frame,game):
        new_effects = []

        for effect in game.miss_effects:
            age = time.time() - effect["time"]

            if age < 0.5:  # duration
                progress = age / 0.5

                # expanding ring
                radius = int(10 + 50 * progress)

                # fading
                alpha = 1 - progress

                overlay = frame.copy()

                cv2.circle(overlay,(effect["x"], effect["y"] + 10),radius,(0, 0, 255),3)

                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                new_effects.append(effect)

        game.miss_effects = new_effects


    def draw_position_line(self, frame, game):
        h, w, _ = frame.shape

        # Same depth as target (very important for alignment)
        line_y = int(game.target_y + 40)

        overlay = frame.copy()

        # Thick glowing line
        #cv2.line(overlay, (50, line_y), (w - 50, line_y), (255, 255, 0), 8)

        # Main perspective line (front)
        cv2.line(overlay, (100, line_y), (w - 100, line_y), (255, 255, 0), 6)

        # Secondary faint line (depth illusion)
        cv2.line(overlay, (120, line_y + 5), (w - 120, line_y + 5), (200, 200, 0), 2)

        # Blend for glow effect
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Instruction text
        cv2.putText(frame,"Stand behind this line",(int(w * 0.25), line_y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0),2)

    

