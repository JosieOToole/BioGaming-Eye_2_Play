# BioGaming-Eye_2_Play
IMU and IR camera Minecraft head and eye controller
===================================================

Files
-----
minecraft_controller.ino     : upload to the Arduino Leonardo
minecraft_eye_controller.py  : run on the laptop

Use BOTH files together. Keep Arduino Serial Monitor and Serial Plotter CLOSED while
the Python program is running; the laptop uses the Leonardo USB port.

Install and launch
------------------
1. In Arduino IDE, select Arduino Leonardo and its current USB port.
   Install the "SparkFun BNO080 Cortex Based IMU" library and Keyboard 
   and Mouse are the Arduino HID libraries if needed.
   Open minecraft_controller.ino, compile, and upload.
2. In Terminal, install the laptop packages if needed:
    python -m pip install opencv-python numpy pyserial
3. Start the existing Pi camera stream with: rpicam-vid -t 0 --nopreview --inline --listen 
   --width 640 --height 480 --framerate 15 --autofocus-mode manual --lens-position 10 -o tcp://0.0.0.0:8888
4. In the folder containing minecraft_eye_controller.py, find the port:
   python .\minecraft_eye_controller.py --list-ports
   Then run with the port actually listed, for example:
   python .\minecraft_eye_controller.py --port COM14
   For another camera address:
   python .\minecraft_eye_controller.py --port COM14 --source tcp://10.203.125.120:8888
5. Draw the eye box to include the pupil's full travel, then press Enter.

Calibrate BEFORE starting game controls
--------------------------------------
Click the pupil video window. No pupil circle appears until the IMU has 
been calibrated and is neutral.

1. Wear the controller and face the screen.
   Press n and hold a neutral head position for at least 3 seconds.
   The script displays IMU CALIBRATION, then IMU OK. If movement is detected,
   calibration waits for stillness, up to 10 seconds. Press n again if it fails.
   Sometimes you have to click outside the window for the ok to appear.
2. Keep your head in that same neutral position while moving only your eyes.
   Press c and look at the center of your screen. A marker will appear.
   Each capture gives you 1 second to settle and collects eye samples for 1.5 seconds.
3. Repeat with l (left), r (right), u (up), and d (down). Use comfortable
   but distinct markers near the screen, but make sure no markers are too close to 
   each other cause that can lead to accidental movement.
   Wait for each capture to finish. If capture is cancelled because the head
   moved, return to neutral and retry that direction.
4. Confirm the screen shows IMU OK, eyes OK, and all five letters captured
   (the displayed compact indicator is CLRUD).
5. Return your gaze to center. Press g, then focus on Minecraft within 3 seconds,
   keeping your head neutral. Starting DOES NOT recenter or erase calibration.
   If start is rejected, return to neutral, wait briefly, and press g again.

During play
-----------
- Head tilt: WASD. Tilting forward presses is W, to the left presses A,
  backward presses S, and to the right presses D until you return to neutral
  or change directions. Jumping is flicking the head upward during a forward
  movement. Sprinting is flicking your head from neutral to forward at a
  faster speed and a bigger angle from neutral.
- Eye direction: USB mouse movement at a relatively slow rate, ONLY when the IMU 
  is neutral. Eye movement up moves the camera up, left moves the camera left,
  backward moves the camera backward, and right moves the camera right.
- During a head tilt, movement, sprint flick, or jump, eye detection
  pauses and the Arduino blocks all gaze-driven mouse motion.
- Eye tracking resumes after the head has been settled in neutral for 250 ms.
  Old gaze history is cleared; the five calibration positions are kept.
- Neutral uses the existing WASD dead zone and hysteresis. After walking,
  return far enough to release the movement key, not just below its start angle.
- Mouse speed is initially 3 counts per 20 ms in either axis.

Pause and recalibrate
---------------------
x or Escape : pause ALL game controls; eye/IMU calibrations stay saved
              (click the video window first). g starts another countdown.
n           : recapture IMU neutral with controls OFF; saved eye points are kept
              Wait for IMU OK, then press g. No eye recalibration is required
              if the camera mounting and your viewing position are unchanged.
c           : recapture eye center with controls OFF; then redo l/r/u/d
o           : redraw the eye box with controls OFF; then redo eye calibration
v           : record / stop recording the raw eye video
q           : stop controls and close the laptop program

Any eye calibration capture pauses game controls first. Changing the bridge,
roundness, or darkness detection settings clears eye calibration and pauses
controls. If the camera mounting or viewing position changes, redo eye calibration.
If only the IMU neutral drifts, press n; the saved eye points stay intact.
Pausing / starting alone does not require recalibration.

Connection behavior
-------------------
A background communication thread sends connection checks and reads serial
replies independently of camera frame reads. The Controller line and terminal retain
stop, reset, and start-rejection messages. The laptop and Arduino exchange status
messages and connection checks. If the Arduino receives no connection check for 500 ms,
all controls stop. If no gaze command arrives for 250 ms, the mouse stops.
Outdated IMU readings block mouse movement on the Arduino's next loop.
Controls stop if orientation readings time out. If the stream ends and Python exits,
all controls stop. If a camera read stalls while Python remains open, IMU controls stay
connected and the mouse stops after its gaze commands time out.
