import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

FOV_X_DEG = 122.0  # approximate horizontal field-of-view of camera

# Depth ordering:
# camera (0) < SCREEN_Z < FINGER_Z < EYE_Z
SCREEN_Z = 0.10   # depth of the laptop screen plane
FINGER_Z = 0.70   # depth of fingertip
EYE_Z    = 0.90   # depth of eyes

# fallback (without calibration)
SCREEN_W = 1.45   # fallback physical width of the screen plane
SCREEN_H = 0.8    # fallback height of the screen plane

# Mediapipe landmark indices
LEFT_IRIS = 468
RIGHT_IRIS = 473  # we'll just use the right eye as gaze origin

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

WINDOW_NAME = "Pointing calibration"

def compute_intrinsics(width: int, height: int):
    """Approximate camera intrinsics from image size and a chosen FOV."""
    cx = width / 2.0
    cy = height / 2.0
    fx = width / (2.0 * np.tan(0.5 * np.deg2rad(FOV_X_DEG)))
    fy = fx
    return fx, fy, cx, cy

def unproject(u: float, v: float, Z: float, fx: float, fy: float, cx: float, cy: float):
    """
    Back-project pixel (u, v) to a 3D point at depth Z in camera coordinates.

    Pinhole model: u = fx * X/Z + cx  => X = (u - cx) * Z / fx
                  v = fy * Y/Z + cy  => Y = (v - cy) * Z / fy
    """
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)


def project(point: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """Project a 3D camera-space point back to image pixels."""
    X, Y, Z = point
    if Z == 0:
        return None
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.array([u, v], dtype=np.float32)


def intersect_ray_with_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, plane_z: float):
    """
    Intersect a ray with the plane z = plane_z in camera coordinates.

    ray(t) = ray_origin + t * ray_dir
    plane: Z = plane_z
    """
    dz = ray_dir[2]
    if abs(dz) < 1e-6:
        return None  # ray parallel to plane

    t = (plane_z - ray_origin[2]) / dz
    if t <= 0:
        return None  # intersection is behind the eye or not between eye and screen

    return ray_origin + t * ray_dir


def draw_virtual_screen(frame, fx, fy, cx, cy, screen_corners_3d: Optional[np.ndarray] = None):
    """
    Draw the virtual screen plane as a projected quadrilateral with diagonals.
    If screen_corners_3d is None, use a default centered rectangle on z = SCREEN_Z.
    NOTE: This is just for visualization in the camera image.
    """
    if screen_corners_3d is None:
        # Fallback symmetric rectangle
        corners_3d = [
            np.array([-SCREEN_W / 2, -SCREEN_H / 2, SCREEN_Z], dtype=np.float32),  # TL
            np.array([ SCREEN_W / 2, -SCREEN_H / 2, SCREEN_Z], dtype=np.float32),  # TR
            np.array([ SCREEN_W / 2,  SCREEN_H / 2, SCREEN_Z], dtype=np.float32),  # BR
            np.array([-SCREEN_W / 2,  SCREEN_H / 2, SCREEN_Z], dtype=np.float32),  # BL
        ]
    else:
        corners_3d = screen_corners_3d

    corners_2d = []
    for p in corners_3d:
        uv = project(p, fx, fy, cx, cy)
        if uv is None:
            return
        corners_2d.append(tuple(uv.astype(int)))

    color = (255, 255, 0)
    # rectangle
    for i in range(4):
        cv2.line(frame, corners_2d[i], corners_2d[(i + 1) % 4], color, 2)
    # diagonals
    cv2.line(frame, corners_2d[0], corners_2d[2], color, 1)
    cv2.line(frame, corners_2d[1], corners_2d[3], color, 1)


def screen_uv_from_hit(hit_3d: np.ndarray, screen_corners_3d: Optional[np.ndarray]):
    """
    Map a hit point on the screen plane to (u, v) in [0,1]x[0,1].

    If screen_corners_3d is provided, treat TL, TR, BR, BL as defining a
    (possibly skewed) quad and solve P ≈ TL + u*(TR-TL) + v*(BL-TL) via
    least-squares. Otherwise fall back to centered rectangle math.
    """
    X, Y, Z = hit_3d

    if screen_corners_3d is None:
        # Fallback mapping for centered fronto-parallel rectangle
        u = (X + SCREEN_W / 2) / SCREEN_W
        v = (Y + SCREEN_H / 2) / SCREEN_H
        return u, v

    TL, TR, BR, BL = screen_corners_3d
    origin = TL
    U = TR - TL  # horizontal direction
    V = BL - TL  # vertical direction

    rhs = hit_3d - origin  # 3D vector
    M = np.stack([U, V], axis=1)  # shape (3, 2)

    # Least-squares solution to M @ [u, v] ≈ rhs
    uv, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    u, v = uv[0], uv[1]
    return float(u), float(v)

def detect_eye_and_finger(frame, face_mesh, hands):
    """
    Run MediaPipe face mesh + hands on the given frame and return:
    - eye_px: (x, y) of gaze origin (right eye)
    - finger_px: (x, y) of index fingertip
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    eye_px = None
    finger_px = None

    # --- Eye center: single right eye ---
    if face_results.multi_face_landmarks:
        lms = face_results.multi_face_landmarks[0].landmark
        eye = lms[RIGHT_IRIS]
        ex = eye.x * w
        ey = eye.y * h
        eye_px = np.array([ex, ey], dtype=np.float32)

    # --- Index fingertip ---
    if hand_results.multi_hand_landmarks:
        hlms = hand_results.multi_hand_landmarks[0].landmark
        tip = hlms[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        fx_px = tip.x * w
        fy_px = tip.y * h
        finger_px = np.array([fx_px, fy_px], dtype=np.float32)

    return eye_px, finger_px

def calibrate_screen_by_pointing(cap, face_mesh, hands, fx, fy, cx, cy):
    """
    Ask the user to point at TL, TR, BR, BL corners of the *computer screen*.
    For each, intersect the eye->finger ray with plane z = SCREEN_Z and
    record the resulting 3D points as screen corners.

    NOTE: This shows its own OpenCV window and waits for 'c' / 'q'.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    

    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    corners_3d = []

    # where to draw the red target circle (in camera pixels) for each label
    def corner_target(label: str, w: int, h: int) -> Tuple[int, int]:
        # very small margin so it’s almost exactly at the image corner
        margin_x = int(0.03 * w)
        margin_y = int(0.03 * h)
        if label == "TOP-LEFT":
            return margin_x, margin_y
        elif label == "TOP-RIGHT":
            return w - margin_x, margin_y
        elif label == "BOTTOM-RIGHT":
            return w - margin_x, h - margin_y
        elif label == "BOTTOM-LEFT":
            return margin_x, h - margin_y
        else:
            return w // 2, h // 2

    for label in labels:
        print(f"Calibration: point at {label} corner of the SCREEN and press 'c' to capture.")

        hit_for_this_corner = None

        while True:
            ret, frame = cap.read()
            if not ret:
                cv2.destroyWindow(WINDOW_NAME)
                return None
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            eye_px, finger_px = detect_eye_and_finger(frame, face_mesh, hands)

            # --- draw the red target circle for this corner (no big blue box) ---
            tx, ty = corner_target(label, w, h)
            cv2.circle(frame, (tx, ty), 16, (0, 0, 255), -1)  # filled red circle
            # --------------------------------------------------------------------

            if eye_px is not None:
                cv2.circle(frame, (int(eye_px[0]), int(eye_px[1])), 5, (255, 0, 0), -1)
            if finger_px is not None:
                cv2.circle(frame, (int(finger_px[0]), int(finger_px[1])), 7, (0, 255, 0), -1)

            if eye_px is not None and finger_px is not None:
                # back-project to 3D
                eye_3d = unproject(eye_px[0], eye_px[1], EYE_Z, fx, fy, cx, cy)
                finger_3d = unproject(finger_px[0], finger_px[1], FINGER_Z, fx, fy, cx, cy)
                ray_origin = eye_3d
                ray_dir = finger_3d - eye_3d

                # 2D line visualization
                cv2.line(
                    frame,
                    (int(eye_px[0]), int(eye_px[1])),
                    (int(finger_px[0]), int(finger_px[1])),
                    (255, 0, 255),
                    2,
                )

                hit_3d = intersect_ray_with_plane(ray_origin, ray_dir, SCREEN_Z)
                if hit_3d is not None:
                    # project this intersection into camera image (just for debugging)
                    hit_px = project(hit_3d, fx, fy, cx, cy)
                    if hit_px is not None:
                        hx, hy = int(hit_px[0]), int(hit_px[1])
                        cv2.circle(frame, (hx, hy), 9, (0, 0, 255), -1)
                        hit_for_this_corner = hit_3d

            cv2.putText(
                frame,
                f"Point at {label} screen corner and press 'c' (or 'q' to skip)",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                # user aborted calibration
                cv2.destroyWindow(WINDOW_NAME)
                return None

            if key == ord("c") and hit_for_this_corner is not None:
                corners_3d.append(hit_for_this_corner)
                break

    cv2.destroyWindow(WINDOW_NAME)

    if len(corners_3d) != 4:
        return None

    return np.array(corners_3d, dtype=np.float32)

class PointingTracker:
    """
    Usage:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        tracker = PointingTracker(cap, w, h)  # runs calibration once

        # In your game loop, for each new frame:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        sx, sy = tracker.process_frame(frame)
        # (sx, sy) are pixel coords in camera image, or (None, None) if not valid.
    """

    def __init__(self, cap: cv2.VideoCapture, width: int, height: int, do_calibration: bool = True):
        self.cap = cap
        self.width = width
        self.height = height

        self.fx, self.fy, self.cx, self.cy = compute_intrinsics(width, height)

        # Create mediapipe models
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Screen geometry from calibration
        if do_calibration:
            self.screen_corners_3d = calibrate_screen_by_pointing(
                self.cap, self.face_mesh, self.hands, self.fx, self.fy, self.cx, self.cy
            )
            if self.screen_corners_3d is None:
                print("Calibration skipped or failed. Using default centered screen.")
        else:
            self.screen_corners_3d = None

    def process_frame(self, frame: np.ndarray, draw_debug: bool = False) -> Tuple[Optional[int], Optional[int]]:
        """
        Given a BGR frame (already flipped horizontally like in calibration),
        return (sx, sy) in camera pixel coordinates where you're pointing,
        or (None, None) if we can't detect eye/finger yet.

        If draw_debug=True, draws some debug overlays on the frame.
        """
        h, w, _ = frame.shape

        eye_px, finger_px = detect_eye_and_finger(frame, self.face_mesh, self.hands)

        if draw_debug:
            # draw calibrated screen quad (if available) or fallback
            draw_virtual_screen(frame, self.fx, self.fy, self.cx, self.cy, self.screen_corners_3d)
            if eye_px is not None:
                cv2.circle(frame, (int(eye_px[0]), int(eye_px[1])), 5, (255, 0, 0), -1)
            if finger_px is not None:
                cv2.circle(frame, (int(finger_px[0]), int(finger_px[1])), 7, (0, 255, 0), -1)

        if eye_px is None or finger_px is None:
            return None, None

        # 3D eye and finger
        eye_3d = unproject(eye_px[0], eye_px[1], EYE_Z, self.fx, self.fy, self.cx, self.cy)
        finger_3d = unproject(finger_px[0], finger_px[1], FINGER_Z, self.fx, self.fy, self.cx, self.cy)

        ray_origin = eye_3d
        ray_dir = finger_3d - eye_3d

        if draw_debug:
            cv2.line(
                frame,
                (int(eye_px[0]), int(eye_px[1])),
                (int(finger_px[0]), int(finger_px[1])),
                (255, 0, 255),
                2,
            )

        hit_3d = intersect_ray_with_plane(ray_origin, ray_dir, SCREEN_Z)
        if hit_3d is None:
            return None, None

        u, v = screen_uv_from_hit(hit_3d, self.screen_corners_3d)

        # Clamp to [0,1]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        sx = int(u * (w - 1))
        sy = int(v * (h - 1))

        if draw_debug:
            cv2.circle(frame, (sx, sy), 9, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Screen uv: {u:.2f}, {v:.2f}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return sx, sy


# Optional: keep a small standalone test if you run pointing.py directly
def _demo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from camera")
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    tracker = PointingTracker(cap, w, h, do_calibration=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        sx, sy = tracker.process_frame(frame, draw_debug=True)
        if sx is not None:
            cv2.circle(frame, (sx, sy), 9, (0, 0, 255), -1)

        cv2.imshow("Pointing demo (press q or ESC)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
