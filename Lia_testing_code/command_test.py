"""
I_2_Play — Test 1 command accuracy logger (no filming needed)

The Leonardo acts as a USB keyboard + mouse, so every command the device makes
reaches the laptop as a key press (W/A/S/D, Space, Ctrl) or mouse movement (eye
commands). This program shows a random cue on screen, records what the device
sends in the next few seconds, and saves the result to a CSV after every trial.

SETUP (once per laptop)
  1. python -m pip install pygame pandas matplotlib scipy
  2. Put command_test.py and analyze_test1.py in one folder, then cd into it.
  3. Plug the Leonardo into THIS laptop and run the eye-tracking program.

STEP 1 - Check signals (once per laptop, before the first participant)
  python command_test.py --check
  Do each command once and confirm the window names it correctly. Esc = close.
  If Look Up / Look Down are swapped, set MOUSE_UP_IS_NEGATIVE_Y = False below.

STEP 2 - Run each participant (P1 to P5, each gets its own random cue order)
  a. In the eye program: n (IMU), c/l/r/u/d (eyes), then g.
  b. During the 3-s countdown, click THIS test window (not Minecraft).
  c. python command_test.py --participant 1
  d. Press P to start. Follow the cue on screen, return to neutral after each.
  e. 100 trials (10 commands x 10 reps, ~10 min), with a break at trial 50.
     When finished, the window shows DONE and closes by itself.
  f. Repeat with --participant 2, 3, 4, 5.

Operator keys (the device never sends these):
  P = start / pause / resume
  X = mark the previous trial invalid (misheard cue) and redo it at the end
  Esc = quit early (all finished trials are already saved)

WHERE FILES ARE SAVED
  results/P1_test1.csv, results/P2_test1.csv, ... in the folder you ran from.
  Choose another folder with: --out C:\\path\\to\\results
  Rerunning the same participant renames the old file to P#_test1_backup_*.csv.
  Using several laptops: copy all P#_test1.csv files into one results folder.

STEP 3 - Analyze (after all participants)
  python analyze_test1.py results
  Writes tables + figures to results/analysis/.
"""
import argparse, csv, os, random, time
import pygame

# ---------------------------------------------------------------- settings
COMMANDS = ["Forward (W)", "Back (S)", "Left (A)", "Right (D)", "Jump", "Sprint",
            "Look Left", "Look Right", "Look Up", "Look Down"]
HEAD = {"Forward (W)", "Back (S)", "Left (A)", "Right (D)", "Jump", "Sprint"}
KEY_TO_CMD = {"W": "Forward (W)", "S": "Back (S)", "A": "Left (A)", "D": "Right (D)"}
REPS = 10            # reps of each command per person (10 x 10 = 100 trials)
RESPONSE_S = 3.0     # time allowed after the cue appears
REST_S = 2.5         # return-to-neutral time; device output here is ignored
MOUSE_THRESHOLD = 15 # mouse counts in one direction before it counts as a gaze command
MOUSE_UP_IS_NEGATIVE_Y = True   # flip if --check shows Look Up/Down reversed
SEED = 2026          # same cue order as the Excel workbook's P1-P5 sheets

KEYS = {pygame.K_w: "W", pygame.K_a: "A", pygame.K_s: "S", pygame.K_d: "D",
        pygame.K_SPACE: "SPACE", pygame.K_LCTRL: "CTRL", pygame.K_RCTRL: "CTRL"}


def cue_order(participant, reps=REPS, seed=SEED):
    """Blocked randomization: each block has every command once, no back-to-back
    repeats. Identical to the order printed in the Excel workbook."""
    rng = random.Random(seed)
    for p in range(1, participant + 1):
        seq, prev = [], None
        for _ in range(reps):
            while True:
                blk = COMMANDS[:]; rng.shuffle(blk)
                if blk[0] != prev: break
            seq += blk; prev = blk[-1]
        if p == participant:
            return seq


def classify(events):
    """events: list of (ms_after_cue, 'key', name) or (ms, 'mouse', (dx, dy)).
    Returns (detected command, response ms, all outputs seen)."""
    keys = [(t, k) for t, kind, k in events if kind == "key"]
    names = [k for _, k in keys]

    # Head: Space = jump; Ctrl together with W = sprint; else first movement key
    head, head_t = None, None
    if "SPACE" in names:
        head, head_t = "Jump", next(t for t, k in keys if k == "SPACE")
    elif "CTRL" in names and "W" in names:
        head, head_t = "Sprint", next(t for t, k in keys if k == "CTRL")
    else:
        for t, k in keys:
            if k in KEY_TO_CMD:
                head, head_t = KEY_TO_CMD[k], t
                break

    # Eye: first moment the summed mouse movement passes the threshold
    eye, eye_t, mx, my = None, None, 0, 0
    for t, kind, v in events:
        if kind != "mouse": continue
        mx += v[0]; my += v[1]
        if max(abs(mx), abs(my)) >= MOUSE_THRESHOLD:
            if abs(mx) >= abs(my):
                eye = "Look Right" if mx > 0 else "Look Left"
            else:
                up = (my < 0) if MOUSE_UP_IS_NEGATIVE_Y else (my > 0)
                eye = "Look Up" if up else "Look Down"
            eye_t = t
            break

    seen = sorted({KEY_TO_CMD.get(k, k) for k in names} | ({eye} if eye else set()))
    if head and eye:   # both happened: whichever came first counts
        det, t = (head, head_t) if head_t <= eye_t else (eye, eye_t)
    elif head or eye:
        det, t = (head, head_t) if head else (eye, eye_t)
    else:
        det, t = "No response", None
    return det, t, seen


# ---------------------------------------------------------------- display
def draw(screen, fonts, big, small="", color=(255, 255, 255), corner=""):
    screen.fill((20, 24, 38))
    w, h = screen.get_size()
    t = fonts[0].render(big, True, color); screen.blit(t, t.get_rect(center=(w // 2, h // 2 - 20)))
    if small:
        s = fonts[1].render(small, True, (170, 180, 200)); screen.blit(s, s.get_rect(center=(w // 2, h // 2 + 70)))
    if corner:
        c = fonts[2].render(corner, True, (120, 130, 150)); screen.blit(c, (15, 12))
    pygame.display.flip()


def read_events():
    """Returns (device events, operator key) since the last call."""
    out, op = [], None
    for e in pygame.event.get():
        if e.type == pygame.QUIT: op = "QUIT"
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: op = "QUIT"
            elif e.key == pygame.K_p: op = "PAUSE"
            elif e.key == pygame.K_x: op = "REDO"
            elif e.key in KEYS: out.append(("key", KEYS[e.key]))
        elif e.type == pygame.MOUSEMOTION and e.rel != (0, 0):
            out.append(("mouse", e.rel))
    return out, op


def setup_window():
    pygame.init()
    screen = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption("I_2_Play — Test 1")
    pygame.event.set_grab(True); pygame.mouse.set_visible(False)   # relative mouse
    fonts = (pygame.font.SysFont("arial", 96, bold=True), pygame.font.SysFont("arial", 30),
             pygame.font.SysFont("arial", 20))
    return screen, fonts


def check_mode():
    """Live view of what the device sends, for testing before the session."""
    screen, fonts = setup_window()
    window, last = [], "Waiting for device output..."
    while True:
        ev, op = read_events()
        if op == "QUIT": break
        now = time.monotonic() * 1000
        window = [(t, k, v) for t, k, v in window if now - t < 800] + [(now, k, v) for k, v in ev]
        if ev:
            det, _, seen = classify([(t - window[0][0], k, v) for t, k, v in window])
            last = f"{det}   (outputs: {', '.join(seen)})"
        draw(screen, fonts, "CHECK MODE", last, corner="Do each command once and confirm it is named correctly.  Esc = quit")
        pygame.time.wait(10)
    pygame.quit()


def run(participant, reps, response_s, rest_s, outdir):
    queue = [(i + 1, c) for i, c in enumerate(cue_order(participant, reps))]
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"P{participant}_test1.csv")
    if os.path.exists(path):   # a restarted session starts fresh; the old file is kept as a backup
        backup = path.replace("_test1.csv", "_test1_backup_" + time.strftime("%H%M%S") + ".csv")
        os.rename(path, backup)
        print("Existing file renamed to", backup)
    f = open(path, "w", newline="")
    wr = csv.writer(f)
    if True:
        wr.writerow(["participant", "trial", "block", "attempt", "cue", "cue_type", "detected",
                     "correct", "response_ms", "all_outputs", "valid", "timestamp"])
    screen, fonts = setup_window()
    n_cmd = len(COMMANDS)
    draw(screen, fonts, f"P{participant}", "Calibrate, press g, then click here.  Press P to start.")
    while True:
        _, op = read_events()
        if op == "PAUSE": break
        if op == "QUIT": pygame.quit(); f.close(); return
        pygame.time.wait(20)

    done, attempts, last_row = 0, {}, None
    i = 0
    while i < len(queue):
        trial, cue = queue[i]
        attempts[trial] = attempts.get(trial, 0) + 1
        corner = f"P{participant}   trial {trial}   ({done}/{len(COMMANDS) * reps} done)   P pause · X redo previous · Esc quit"

        # rest / neutral (device output ignored)
        t_end = time.monotonic() + rest_s
        quit_now = False
        while time.monotonic() < t_end:
            _, op = read_events()
            if op == "QUIT": quit_now = True; break
            if op == "REDO" and last_row is not None:
                last_row[-2] = 0; wr.writerow(last_row); f.flush()   # logged again as invalid
                queue.append((last_row[1], last_row[4])); done -= 1; last_row = None
            if op == "PAUSE":
                draw(screen, fonts, "PAUSED", "Press P to resume", corner=corner)
                while True:
                    _, op2 = read_events()
                    if op2 == "PAUSE": break
                    if op2 == "QUIT": quit_now = True; break
                    pygame.time.wait(20)
                if quit_now: break
                t_end = time.monotonic() + rest_s
            draw(screen, fonts, "neutral", "Head straight · eyes on center", (120, 130, 150), corner)
            pygame.time.wait(10)
        if quit_now: break

        # cue + response window
        events = []
        draw(screen, fonts, cue.upper(), "", (255, 220, 90), corner)
        t0 = time.monotonic()
        while time.monotonic() - t0 < response_s:
            ev, op = read_events()
            now = (time.monotonic() - t0) * 1000
            events += [(now, k, v) for k, v in ev]
            if op == "QUIT": quit_now = True; break
            pygame.time.wait(2)
        if quit_now: break
        det, rt, seen = classify(events)
        row = [participant, trial, (trial - 1) // n_cmd + 1, attempts[trial], cue,
               "Head" if cue in HEAD else "Eye", det, int(det == cue),
               "" if rt is None else round(rt), "; ".join(seen), 1, time.strftime("%Y-%m-%d %H:%M:%S")]
        wr.writerow(row); f.flush()
        last_row, done, i = row, done + 1, i + 1
        if trial == n_cmd * reps // 2 and attempts[trial] == 1:
            draw(screen, fonts, "BREAK", "1-minute break. Press P to continue.", corner=corner)
            while True:
                _, op = read_events()
                if op in ("PAUSE", "QUIT"): break
                pygame.time.wait(20)
            if op == "QUIT": break

    f.close()
    draw(screen, fonts, "DONE", f"Saved {path}")
    pygame.time.wait(1500); pygame.quit()
    print("Saved", os.path.abspath(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", type=int, help="1-5")
    ap.add_argument("--check", action="store_true", help="live view of device output")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--response-s", type=float, default=RESPONSE_S)
    ap.add_argument("--rest-s", type=float, default=REST_S)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.check:
        check_mode()
    elif a.participant:
        os.makedirs(a.out, exist_ok=True)
        run(a.participant, a.reps, a.response_s, a.rest_s, a.out)
    else:
        ap.error("use --check or --participant N")

