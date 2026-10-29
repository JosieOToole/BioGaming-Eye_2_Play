"""
Tracks pupil position, classifies gaze direction, and sends directions
to the Arduino Leonardo over USB.

Uses the default camera address: python eye_tracking_controller.py --port COM12

To use a different camera address:
python eye_tracking_controller.py --port COM12 --source tcp://10.203.125.120:8888

Select a region on screen that includes the pupil's full movement range
after an initial photo is taken.

Calibration keys: c = center, l = left, r = right, u = up, d = down
Recentering gets rid of all the other calibrations.

Other keys: o = select the eye region again, v = start/stop video recording,
q = quit
"""

import os

# Less video buffering to reduce stream delay
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

import time
import argparse
import serial
from serial.tools import list_ports

# Keeps a limited history of measurements
from collections import deque

import cv2
import numpy as np

# Pi camera address at apartment or jhu; --source overrides it
CAMERA_STREAM_URL = "tcp://192.168.4.33:8888"  # "tcp://10.203.125.120:8888"

# Rotates live camera frames as camera is mounted sideways
FRAME_ROTATION = cv2.ROTATE_90_CLOCKWISE

# Looks for three consecutive processed frames before changing direction
DIRECTION_CONFIRM_FRAMES = 3

# Finds the pupil using its shape, size, and previous position
class PupilTracker:
    def __init__(self):
        self.previous_pupil = None
        self.recent_pupil_radii = deque(maxlen=40)
        self.size_change_cnt = 0

    # Clears tracking history
    def reset(self):
        self.previous_pupil = None
        self.recent_pupil_radii.clear()
        self.size_change_cnt = 0

    # Processes one grayscale eye image; returns detections and a mask
    def find(self, grayscale_eye, brightness_threshold, gap_fill_size, minimum_roundness=0.3, auto_threshold_percentile=None):
        eye_height, eye_width = grayscale_eye.shape
        detection_result = {
            "pupil": None,
            "glint": None,
            "blink": False,
            "mask": None,
            "reason": "",
            "dark": 0.0,
            "thresh": brightness_threshold,
            "auto_fallback": False,
        }

        # Finds a possible bright reflection; separates from the dark pupil
        bright_threshold = float(np.clip(np.percentile(grayscale_eye, 99.5), 180, 250))
        _, bright_pixel_mask = cv2.threshold(grayscale_eye, bright_threshold, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_bright_contours = []
        max_reflect_area = 0.01 * eye_width * eye_height

        for contour in bright_contours:
            contour_area = cv2.contourArea(contour)
            if 3 <= contour_area <= max_reflect_area:
                small_bright_contours.append(contour)

        if small_bright_contours:
            largest_bright_contour = max(small_bright_contours, key=cv2.contourArea)
            reflect_measure = cv2.moments(largest_bright_contour)
            if reflect_measure["m00"] > 0:
                reflect_center_x = reflect_measure["m10"] / reflect_measure["m00"]
                reflect_center_y = reflect_measure["m01"] / reflect_measure["m00"]
                detection_result["glint"] = (reflect_center_x, reflect_center_y)

        # Automatic mode adapts the cutoff to the image brightness
        if auto_threshold_percentile is not None:
            brightness_threshold = (int(np.percentile(grayscale_eye, auto_threshold_percentile)) + 6)
            detection_result["thresh"] = brightness_threshold

        # Shows dark pixels as white in the pupil mask
        _, pupil_mask = cv2.threshold(grayscale_eye, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
        white_pixel_count = np.count_nonzero(pupil_mask)
        total_pixel_count = eye_width * eye_height
        dark_pixel_frac = white_pixel_count / total_pixel_count

        # Recovers when a manual cutoff selects more than half the image
        if auto_threshold_percentile is None and dark_pixel_frac > 0.5:
            brightness_threshold = int(np.percentile(grayscale_eye, 8)) + 6
            detection_result["thresh"] = brightness_threshold
            detection_result["auto_fallback"] = True
            _, pupil_mask = cv2.threshold(grayscale_eye, brightness_threshold, 255, cv2.THRESH_BINARY_INV)

        # Fills small gaps so the pupil stays in one piece
        if gap_fill_size > 1:
            gap_fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_fill_size, gap_fill_size))
            pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, gap_fill_kernel)

        # Removes small specks
        cleanup_kernel = np.ones((5, 5), dtype=np.uint8)
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, cleanup_kernel)
        detection_result["mask"] = pupil_mask
        white_pixel_count = np.count_nonzero(pupil_mask)
        total_pixel_count = eye_width * eye_height
        dark_pixel_frac = white_pixel_count / total_pixel_count
        detection_result["dark"] = dark_pixel_frac

        if dark_pixel_frac > 0.5:
            detection_result["blink"] = True
            detection_result["reason"] = (f"Mask {dark_pixel_frac * 100:.0f}% dark: ""threshold too high or eye possibly closed")
            return detection_result

        pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pupil_contour = None
        best_candidate_score = 0.0
        highest_roundness = 0.0
        large_contour_count = 0

        # Rejects small, irregular, elongated, oversized, and border regions
        for contour in pupil_contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < 40:
                continue
            large_contour_count += 1
            contour_perimeter = cv2.arcLength(contour, True)
            if contour_perimeter > 0:
                contour_roundness = (4 * np.pi * contour_area / (contour_perimeter * contour_perimeter))
            else:
                contour_roundness = 0.0
            highest_roundness = max(highest_roundness, contour_roundness)
            if contour_roundness < minimum_roundness:
                continue

            box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
            aspect_ratio = min(box_width, box_height) / max(box_width, box_height)
            maximum_pupil_area = 0.15 * eye_width * eye_height

            if aspect_ratio < 0.38 or contour_area > maximum_pupil_area:
                continue

            touches_border = (box_x <= 1 or box_y <= 1 or box_x + box_width >= eye_width - 1 or box_y + box_height >= eye_height - 1)

            if touches_border:
                continue

            pupil_shape_score = contour_roundness * contour_roundness * contour_area

            # Prefers shapes near the last pupil position
            if self.previous_pupil is not None:
                candidate_moments = cv2.moments(contour)

                if candidate_moments["m00"] > 0:
                    candidate_center_x = (candidate_moments["m10"] / candidate_moments["m00"])
                    candidate_center_y = (candidate_moments["m01"] / candidate_moments["m00"])
                    previous_center_x = self.previous_pupil[0]
                    previous_center_y = self.previous_pupil[1]
                    distance_from_previous = np.hypot(candidate_center_x - previous_center_x, candidate_center_y - previous_center_y)
                    pupil_shape_score /= 1.0 + distance_from_previous / (0.3 * eye_width)

            if pupil_shape_score > best_candidate_score:
                best_pupil_contour = contour
                best_candidate_score = pupil_shape_score

        if best_pupil_contour is None:
            detection_result["blink"] = dark_pixel_frac > 0.25

            if large_contour_count == 0:
                detection_result["reason"] = "No dark region large enough"
            else:
                detection_result["reason"] = (f"No candidate passed all checks. "f"Roundest: {highest_roundness * 100:.0f}%; "f"minimum: {minimum_roundness * 100:.0f}%")
            return detection_result

        # The largest inscribed circle estimates pupil center and radius
        filled_pupil_mask = np.zeros_like(pupil_mask)
        cv2.drawContours(filled_pupil_mask, [best_pupil_contour], -1, 255, -1)
        distance_map = cv2.distanceTransform(filled_pupil_mask, cv2.DIST_L2, 5)
        _, pupil_radius, _, pupil_center = cv2.minMaxLoc(distance_map)

        # Ignores sudden size changes and accepts the new size after seven in a row
        if len(self.recent_pupil_radii) >= 10:
            median_pupil_radius = float(np.median(self.recent_pupil_radii))

            if (pupil_radius > 2.2 * median_pupil_radius or pupil_radius < 0.3 * median_pupil_radius):
                self.size_change_cnt += 1
                if self.size_change_cnt <= 6:
                    detection_result["reason"] = ("Sudden pupil-size change: possible eyelid or shadow")
                    return detection_result
                self.recent_pupil_radii.clear()

        self.size_change_cnt = 0

        # History is updated before the final size check
        self.recent_pupil_radii.append(pupil_radius)

        maximum_pupil_radius = 0.22 * min(eye_width, eye_height)

        if pupil_radius < 3 or pupil_radius > maximum_pupil_radius:
            detection_result["reason"] = "Uncertain pupil size"
            return detection_result

        pupil_center_x, pupil_center_y = pupil_center
        self.previous_pupil = (pupil_center_x, pupil_center_y, pupil_radius)
        detection_result["pupil"] = self.previous_pupil
        return detection_result

# Figures out where you are looking using the five calibrated positions
class GazeDirectionClassifier:
    DIRECTION_NAMES = ("LEFT", "RIGHT", "UP", "DOWN")

    def __init__(self):
        self.center_position = None
        self.direction_positions = {}
        self.recent_positions = deque(maxlen=20)
        self.smoothed_position = None
        self.active_direction = set()
        self.pending_direction, self.direction_frame_cnt = (set(), 0)
        self.status = "NOT CALIBRATED"
        self.message = ""
        self.pupil_detected = False
        self.cal_direction_movement = {}

    # Saves the middle position from the samples and checks target spacing
    def capture(self, target_name):
        if not self.pupil_detected or len(self.recent_positions) < 10:
            self.message = "Hold still with pupil visible for at least 10 frames"
            return False
        calibration_samples = np.array(self.recent_positions)

        # Uses the middle value so a few unusual samples have less effect
        calibration_position = np.median(calibration_samples, axis=0)
        cal_movement = float(np.percentile(np.linalg.norm(calibration_samples - calibration_position, axis=1), 90))
        calibration_points = dict(self.direction_positions)
        if self.center_position is not None:
            calibration_points["CENTER"] = self.center_position
        for direction_name, calibration_point in calibration_points.items():
            min_target_space = max(2.0, min(3.0, 1.5 * max(cal_movement, self.cal_direction_movement.get(direction_name, 0))))
            if (direction_name != target_name and np.linalg.norm(calibration_position - calibration_point) < min_target_space):
                self.message = ("Too close to " + direction_name + ": look farther or check marker")
                return False
        if target_name == "CENTER":
            self.center_position = calibration_position
            self.direction_positions.clear()
            self.cal_direction_movement.clear()
        else:
            if self.center_position is None:
                self.message = "Capture CENTER first"
                return False
            self.direction_positions[target_name] = calibration_position
        self.cal_direction_movement[target_name] = cal_movement
        self.active_direction, self.pending_direction, self.direction_frame_cnt = (set(), set(), 0)
        self.message = ("captured " + target_name + (" (movement warning)" 
            if cal_movement > 4 
            else "")
        )
        print(target_name, "xy=", calibration_position.round(2), "jitter=", round(cal_movement, 2))
        return True

    # Clears calibration and direction history
    def reset_calibration(self):
        self.__init__()

    # Smooths pupil movement and checks that the direction is the same
    def update(self, pupil_position, possible_blink, activation_percent, eye_width, eye_height):
        self.pupil_detected = pupil_position is not None and (not possible_blink)
        if not self.pupil_detected:
            self.recent_positions.clear()
            self.smoothed_position = None
            (self.active_direction, self.pending_direction, self.direction_frame_cnt) = (set(), set(), 0)
            self.status = "TRACKING LOST"
            return self.active_direction
        self.recent_positions.append(pupil_position.copy())
        self.smoothed_position = (pupil_position
            if self.smoothed_position is None
            else 0.4 * pupil_position + 0.6 * self.smoothed_position
        )
        if self.center_position is None or len(self.direction_positions) != 4:
            self.active_direction = set()
            self.status = "NOT CALIBRATED"
            return self.active_direction
        direction_names = ["CENTER"] + list(self.DIRECTION_NAMES)
        calibration_points = [self.center_position]
        for direction_name in self.DIRECTION_NAMES:
            calibration_points.append(self.direction_positions[direction_name])
        calibration_points = np.array(calibration_points)
        
        # Considers different left/right and up/down movement ranges
        movement_cal_range = np.maximum(np.ptp(calibration_points, axis=0), 4.0)
        distance_to_target = np.linalg.norm((calibration_points - self.smoothed_position) / movement_cal_range, axis=1)
        closest_target_index = int(np.argmin(distance_to_target))
        possible_direction = set()
        if closest_target_index:
            direction_name = direction_names[closest_target_index]
            target_offset = (calibration_points[closest_target_index] - self.center_position) / movement_cal_range
            offset_from_center = (self.smoothed_position - self.center_position) / movement_cal_range
            move_toward_target = float(offset_from_center @ target_offset) / float(target_offset @ target_offset)
            activation_threshold = max(0.15, activation_percent / 100.0)
            if direction_name in self.active_direction:
                activation_threshold = max(0.1, activation_threshold - 0.1)
            if move_toward_target >= activation_threshold:
                possible_direction = {direction_name}

        # Waits for the same direction across multiple frames before changing it
        if possible_direction == self.pending_direction:
            self.direction_frame_cnt += 1
        else:
            self.pending_direction, self.direction_frame_cnt = (possible_direction, 1)
        if self.direction_frame_cnt >= DIRECTION_CONFIRM_FRAMES:
            self.active_direction = possible_direction
        self.status = "READY"
        return self.active_direction

# Returns the direction name, or CENTER if no direction is active
def get_direction_label(active_direction):
    return next(iter(active_direction), "CENTER")

# Lets you draw a box around the eye that includes its full movement range
def select_eye(frame):
    win_title = "Select eye INCLUDING all pupil travel; Enter"
    eye_region = tuple(map(int, cv2.selectROI(win_title, frame)))
    cv2.destroyWindow(win_title)
    if min(eye_region[2:]) < 20:
        raise ValueError("Eye selection cancelled or too small")
    return (eye_region, None)

# Sends directions to the Leonardo and reads its replies over USB
class LeonardoConnection:
    CODES = {"LEFT": "L", "RIGHT": "R", "UP": "U", "DOWN": "D", "CENTER": "C"}

    def __init__(self, port):
        self.serial_link = serial.Serial(port, 115200, timeout=0, write_timeout=0.1)
        self.last_command = None
        self.last_send_time = 0
        self.last_reply = "waiting"
        self.reply_buffer = bytearray()
        self.send("STOP", send_now=True)

    # Sends when the direction changes or every 0.1 seconds, then checks for replies
    def send(self, direction_label, send_now=False):
        # Sends STOP if the label is not a direction or CENTER
        command_code = self.CODES.get(direction_label, "S")
        current_time = time.monotonic()
        if (send_now or command_code != self.last_command or current_time - self.last_send_time >= 0.1):
            self.serial_link.write((command_code + "\n").encode("ascii"))
            self.last_command = command_code
            self.last_send_time = current_time
        incoming_bytes = self.serial_link.read(min(self.serial_link.in_waiting, 512))
        self.reply_buffer.extend(incoming_bytes)
        while b"\n" in self.reply_buffer:
            reply_line, _, remaining_bytes = self.reply_buffer.partition(b"\n")
            self.reply_buffer = bytearray(remaining_bytes)
            self.last_reply = reply_line.decode("ascii", errors="replace").strip()
        if len(self.reply_buffer) > 512:
            self.reply_buffer.clear()

    # Sends STOP before closing the USB connection
    def close(self):
        try:
            self.serial_link.write(b"S\n")
        finally:
            self.serial_link.close()

# Opens the camera and USB connections and runs eye tracking
def main():
    # Reads the port and optional camera address from the terminal
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", help="Leonardo USB port, e.g. COM5")
    arg_parser.add_argument("--list-ports", action="store_true")
    arg_parser.add_argument("--source", default=CAMERA_STREAM_URL)
    arguments = arg_parser.parse_args()
    if arguments.list_ports:
        for serial_port in list_ports.comports():
            print(serial_port.device, "-", serial_port.description)
        return
    if not arguments.port:
        arg_parser.error("Use --list-ports, then supply --port COMx")
    video_source = arguments.source
    is_live_stream = video_source.startswith("tcp")
    video_capture = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    frame_available, frame = video_capture.read()
    if not frame_available:
        raise SystemExit("Cannot read stream; check Pi streaming command")
    if is_live_stream and FRAME_ROTATION is not None:
        frame = cv2.rotate(frame, FRAME_ROTATION)
    try:
        eye_region, unused_ref_val = select_eye(frame)
    except ValueError as error:
        video_capture.release()
        raise SystemExit(str(error))
    cv2.namedWindow("pupil")
    for direction_name, value, maximum in [("bridge", 5, 21), ("round %", 30, 90), ("dark %", 0, 40), ("enter %", 35, 100)]:
        cv2.createTrackbar(direction_name, "pupil", value, maximum, lambda slider_value: None)
    pupil_tracker = PupilTracker()
    gaze_classifier = GazeDirectionClassifier()
    calibration_in_progress = None
    calibration_samples = []
    status_message = "c/l/r/u/d: timed capture"
    video_writer = None
    previous_detection_settings = None
    previous_direction_label = None
    leonardo = None
    try:
        leonardo = LeonardoConnection(arguments.port)
        while True:
            frame_available, frame = video_capture.read()
            if not frame_available:
                break
            if is_live_stream and FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)
            if video_writer is not None:
                video_writer.write(frame)
            current_time = time.monotonic()
            cam_frame, cam_pos_msg = (frame, "Fixed image reference: recalibrate after camera moves")

            # Changing detection settings clears the saved calibration
            slider_values = []
            for slider_name in ("bridge", "round %", "dark %"):
                slider_values.append(cv2.getTrackbarPos(slider_name, "pupil"))
            detection_settings = tuple(slider_values)
            if (previous_detection_settings is not None and detection_settings != previous_detection_settings):
                gaze_classifier.reset_calibration()
                pupil_tracker.reset()
                calibration_in_progress = None
                status_message = "Detection changed: recalibrate"
            previous_detection_settings = detection_settings
            eye_left, eye_top, eye_width, eye_height = eye_region
            pupil_position = None
            possible_blink = False
            if cam_frame is not None:
                eye_image = cam_frame[eye_top : eye_top + eye_height, eye_left : eye_left + eye_width]
                grayscale_eye = cv2.GaussianBlur(cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
                (gap_fill_size, roundness_percent, darkness_percentile) = detection_settings
                detection_result = pupil_tracker.find(
                    grayscale_eye=grayscale_eye,
                    brightness_threshold=40,
                    gap_fill_size=gap_fill_size | 1,
                    minimum_roundness=max(0.1, roundness_percent / 100),
                    auto_threshold_percentile=max(1, darkness_percentile),
                )
                if detection_result["pupil"] is not None:
                    pupil_position = np.array(detection_result["pupil"][:2], float)
                possible_blink = detection_result["blink"]
                annotated_eye = eye_image.copy()
                mask = detection_result["mask"]
                if pupil_position is not None:
                    pupil_center_x, pupil_center_y, pupil_radius = detection_result["pupil"]
                    cv2.circle(annotated_eye, (int(pupil_center_x), int(pupil_center_y)), int(pupil_radius), (0, 255, 0), 2)
                detection_reason = detection_result["reason"]
            else:
                annotated_eye = frame[eye_top : eye_top + eye_height, eye_left : eye_left + eye_width].copy()
                mask = np.zeros((eye_height, eye_width), np.uint8)
                detection_reason = cam_pos_msg
                pupil_tracker.reset()
            
            # Uses the saved calibration to figure out the gaze direction
            active_direction = gaze_classifier.update(
                pupil_position,
                possible_blink,
                max(15, cv2.getTrackbarPos("enter %", "pupil")),
                eye_width,
                eye_height,
            )
            direction_label = (gaze_classifier.status
                if gaze_classifier.status != "READY"
                else get_direction_label(active_direction)
            )
            if cam_frame is None:
                direction_label = "REFERENCE LOST"

            # Gives you one second to settle, then collects samples for 1.5 seconds
            if calibration_in_progress:
                direction_name, capture_start_time = calibration_in_progress
                capture_elapsed_time = current_time - capture_start_time
                direction_label = "CALIBRATING " + direction_name
                status_message = (f"Hold {direction_name}: {max(0, 2.5 - capture_elapsed_time):.1f}s")
                if (capture_elapsed_time >= 1 and pupil_position is not None and (not possible_blink)):
                    calibration_samples.append(pupil_position.copy())
                if capture_elapsed_time >= 2.5:
                    if len(calibration_samples) >= 8:
                        gaze_classifier.recent_positions = deque(calibration_samples, maxlen=len(calibration_samples))
                        gaze_classifier.pupil_detected = True
                        gaze_classifier.capture(direction_name)
                        status_message = gaze_classifier.message
                    else:
                        status_message = "Too few valid samples: retry this direction"
                    calibration_in_progress = None
                    calibration_samples = []

            # Sends STOP during calibration or when tracking is not ready
            leonardo.send(direction_label)
            if direction_label != previous_direction_label:
                print(direction_label)
                previous_direction_label = direction_label
            calibration_points = {"C": gaze_classifier.center_position}
            for (direction_name, calibration_point) in gaze_classifier.direction_positions.items():
                calibration_points[direction_name[0]] = calibration_point
            if cam_frame is not None:
                for direction_name, cal_position in calibration_points.items():
                    if cal_position is not None:
                        marker_x, marker_y = map(int, cal_position)
                        cv2.drawMarker(annotated_eye, (marker_x, marker_y), (255, 0, 255), cv2.MARKER_CROSS, 8, 1)
                        cv2.putText(annotated_eye, direction_name, (marker_x + 4, marker_y - 4), 0, 0.4, (255, 0, 255), 1)

            # Shows the eye image beside the black-and-white pupil mask
            image_display = np.hstack((annotated_eye, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
            image_display = cv2.resize(image_display, (max(900, 2 * eye_width), int(eye_height * max(900, 2 * eye_width) / (2 * eye_width))))
            status_panel = np.zeros((145, image_display.shape[1], 3), np.uint8)
            calibration_letters = ""
            calibration_targets = [("CENTER", gaze_classifier.center_position)]
            for direction_name in gaze_classifier.DIRECTION_NAMES:
                calibration_targets.append((direction_name, gaze_classifier.direction_positions.get(direction_name)))
            for direction_name, calibration_point in calibration_targets:
                if calibration_point is not None:
                    calibration_letters += direction_name[0]
                else:
                    calibration_letters += "-"
            for line_idx, display_text in enumerate(
                [
                    direction_label,
                    f"cal {calibration_letters} | {cam_pos_msg}",
                    status_message,
                    detection_reason + " | USB: " + leonardo.last_reply,
                ]
            ):
                cv2.putText(
                    status_panel,
                    display_text,
                    (10, 30 + 32 * line_idx),
                    0,
                    0.65,
                    (0, 255, 0) if line_idx == 0 else (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("pupil", np.vstack((status_panel, image_display)))

            # Click the video window first so it receives these key presses
            pressed_key = cv2.waitKey(1) & 255
            cal_keys = {ord("c"): "CENTER", ord("l"): "LEFT", ord("r"): "RIGHT", ord("u"): "UP", ord("d"): "DOWN"}
            if pressed_key in cal_keys:
                leonardo.send("STOP", send_now=True)
                calibration_in_progress = (cal_keys[pressed_key], time.monotonic())
                calibration_samples = []
            elif pressed_key == ord("o"):
                leonardo.send("STOP", send_now=True)
                try:
                    eye_region, unused_ref_val = select_eye(frame)
                except ValueError as error:
                    status_message = str(error)
                gaze_classifier.reset_calibration()
                pupil_tracker.reset()
                calibration_in_progress = None
            elif pressed_key == ord("v"):
                if video_writer is None:
                    frame_height, frame_width = frame.shape[:2]
                    recording_frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
                    if not 1 <= recording_frame_rate <= 120:
                        recording_frame_rate = 30
                    video_writer = cv2.VideoWriter("raw_eye.mp4", cv2.VideoWriter_fourcc(*"mp4v"), recording_frame_rate, (frame_width, frame_height))
                    if not video_writer.isOpened():
                        video_writer.release()
                        video_writer = None
                        status_message = "Recording failed"
                    else:
                        status_message = "Recording raw_eye.mp4; v stops"
                else:
                    video_writer.release()
                    video_writer = None
                    status_message = "Saved raw_eye.mp4"
            elif pressed_key == ord("q"):
                break
    finally:
        if leonardo is not None:
            leonardo.close()
        if video_writer is not None:
            video_writer.release()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
