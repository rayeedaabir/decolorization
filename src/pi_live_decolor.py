"""
pi_live_decolor.py  —  live decolorization demo on a Raspberry Pi 5
===================================================================
One RGB camera in, several decolorizers side by side, with per-frame timing recorded
so the FPS claim in the paper is backed by measurements rather than an estimate.

Adapted from the AIRC-Fusion Pi demo: same threaded-capture pattern, number-key model
switching, mp4 + CSV logging. Simpler here — this task needs ONE camera, so all of the
dual-camera alignment machinery is gone.

Runs on the Pi AND on an ordinary desktop -- Picamera2 is imported only for `--cam csi:N`,
so a USB webcam needs nothing but numpy + opencv. Debugging the demo on a desktop first
separates camera problems from Pi problems, which is worth doing before deployment day.

    python pi_live_decolor.py --cam usb:0
    python pi_live_decolor.py --cam usb:0 --record --proc 160
    python pi_live_decolor.py --cam usb:0 --record --snap-every 90     # video + auto stills

The selector runs on the CPU through pi_decolor.Decolorizer -- there is no GPU path, and
that is the point: the demo shows the model running where a GPU is not available. This is
the 4-cue model (RGB + opponent-chroma); the OURS overlay shows the four weights it chose.

Controls
  1  BT.709 (fixed standard)     2  Ours (ViT selection)     3  both, side by side
  4  all 16 prototypes live, the selected one highlighted -- adaptivity made visible
  p  save a snapshot SET (input + each method pane, banner-free, + a composite) to --snap-dir
  q  quit (writes demo_decolor.mp4 + demo_fps_log.csv if --record)

Needs only numpy + opencv on the Pi: the selector runs through pi_decolor.Decolorizer.
Pair the CSV with retime_demo_video.py to produce a true-wall-clock playback video.
"""
import argparse, csv, os, threading, time
import cv2
import numpy as np


# --------------------------------------------------------------- camera threads
class USBCameraThread(threading.Thread):
    """Grab frames in the background so a slow model never stalls capture.

    Backend order matters on Windows: the default MSMF path often opens a UVC webcam
    (the Brio included) and then returns nothing, or takes several seconds per frame.
    DirectShow is the reliable one there. On Linux the default V4L2 path is correct, so
    this only reorders when it has to.
    """
    def __init__(self, index=0, resolution=(640, 480), backend=None):
        super().__init__(daemon=True)
        if backend is not None:
            order = [backend]
        elif os.name == "nt":
            order = [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), None]
        else:
            order = [None]

        self.cap, self.backend = None, None
        for b in order:
            cap = cv2.VideoCapture(index) if b is None else cv2.VideoCapture(index, b)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            ok, _ = cap.read()
            if ok:
                self.cap, self.backend = cap, b
                break
            cap.release()
        if self.cap is None:
            avail = []
            for i in range(6):
                for b in order:
                    c = cv2.VideoCapture(i) if b is None else cv2.VideoCapture(i, b)
                    if c.read()[0]:
                        avail.append(f"usb:{i}")
                    c.release()
                    if avail and avail[-1] == f"usb:{i}":
                        break
            raise SystemExit(
                f"could not open camera index {index}.\n"
                + (f"These indices DO work: {', '.join(avail)} — pass one with --cam\n"
                   if avail else "No camera index 0-5 returned a frame. Close anything "
                                 "else using the webcam (Teams, Zoom, the Camera app) "
                                 "and try again.\n"))
        self.frame, self.running = None, True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                time.sleep(0.005)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


class CSICameraThread(threading.Thread):
    """Picamera2 path, for the Camera Module v2 on a CSI port."""
    def __init__(self, camera_num=0, resolution=(640, 480)):
        super().__init__(daemon=True)
        from picamera2 import Picamera2
        self.picam = Picamera2(camera_num)
        cfg = self.picam.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"})
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(1.0)
        self.frame, self.running = None, True

    def run(self):
        while self.running:
            self.frame = self.picam.capture_array()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.picam.stop()


def open_camera(spec, resolution, backend=None):
    kind, _, num = spec.partition(":")
    num = int(num or 0)
    cam = (CSICameraThread(num, resolution) if kind == "csi"
           else USBCameraThread(num, resolution, backend))
    cam.start()
    for _ in range(50):                      # wait for the first frame
        if cam.read() is not None:
            return cam
        time.sleep(0.1)
    raise SystemExit(f"no frames from {spec} — run pi_camera_test.py to find the right port")


# --------------------------------------------------------------- display helper
def gui_available():
    """Does this OpenCV build have a GUI?

    `opencv-python-headless` has no highgui, so cv2.imshow raises. It is easy to end up
    with it without asking: pyiqa (used for the NIQE/BRISQUE runs) depends on it, and
    since both packages install the same `cv2` module, whichever landed last wins.
    Probing once at startup beats crashing several hundred frames in.
    """
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def w4(wsel):
    """The 4 cue weights to display (R, G, B, CHROMA).

    The RGB+chroma codebook stores 5-vectors [R, G, B, MSR, CHROMA] with the MSR column
    always 0 (MSR is an ablation-only cue, not part of the 4-cue model). Show the four
    non-MSR weights so the overlay matches the paper's Delta^3 weighting; if a true
    4-column codebook is ever used, show it as-is.
    """
    wsel = np.asarray(wsel).ravel()
    if wsel.size == 5:
        wsel = wsel[[0, 1, 2, 4]]
    return "  ".join(f"{v:.2f}" for v in wsel)


def save_snapshot(clean, view, outdir):
    """Write a publication-ready snapshot SET: each pane banner-free, plus the composite.

    `clean` maps a name (input/bt709/ours/rival/grid) to its BGR image without any overlay,
    so the saved panes drop straight into the paper. All files in one set share a timestamp.
    """
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000) % 1000:03d}"
    saved = []
    for name, img in clean.items():
        fn = os.path.join(outdir, f"{ts}_{name}.png")
        cv2.imwrite(fn, img)
        saved.append(fn)
    comp = os.path.join(outdir, f"{ts}_composite.png")
    cv2.imwrite(comp, view)
    saved.append(comp)
    print(f"saved snapshot set -> {outdir}/  ({len(saved)} files, stamp {ts})")
    return saved


def _fit(text, width, scale, thick=1, floor=0.30):
    """Largest scale <= `scale` at which `text` fits `width` px. Never truncate a label.

    The old banner drew at a fixed 0.55/0.42 regardless of pane width. At --disp 256 the
    OURS label ("OURS P11 w=(0.87 0.05 0.03 0.04 0.02)") is wider than the pane, so it ran
    into the neighbouring pane and was clipped when the strip was later split into tiles.
    """
    while scale > floor:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if w <= width:
            break
        scale -= 0.02
    return scale


def banner(img, text, sub="", sub2=""):
    lines = 1 + bool(sub) + bool(sub2)
    h = (26, 42, 58)[lines - 1]
    avail = img.shape[1] - 16
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                _fit(text, avail, 0.55), (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    _fit(sub, avail, 0.42), (170, 220, 255), 1, cv2.LINE_AA)
    if sub2:
        cv2.putText(img, sub2, (8, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    _fit(sub2, avail, 0.40), (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="usb:0", help="usb:N or csi:N")
    ap.add_argument("--backend", type=int, default=None,
                    help="force an OpenCV capture backend (e.g. 700 = CAP_DSHOW)")
    ap.add_argument("--weights", default="fusion_vit_np.npz")
    ap.add_argument("--proc", type=int, default=160, help="processing resolution")
    ap.add_argument("--disp", type=int, default=320, help="display size per pane")
    ap.add_argument("--grid-cell", type=int, default=0,
                    help="tile size for mode 4 (default: --disp // 2)")
    ap.add_argument("--width", type=int, default=640, help="capture width")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="auto-save a snapshot SET every N frames (0 = only on 'p')")
    ap.add_argument("--snap-dir", default="snaps", help="folder for snapshot sets")
    ap.add_argument("--headless", action="store_true", help="no window; benchmark only")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames")
    ap.add_argument("--metrics", action="store_true",
                    help="live E-score per pane (costs frame rate; see --metric-every)")
    ap.add_argument("--metric-every", type=int, default=5,
                    help="recompute metrics every N frames and hold the value between")
    ap.add_argument("--rival", default="", help="extra method pane, e.g. cv2decolor")
    ap.add_argument("--mode", type=int, default=3, choices=[1, 2, 3, 4],
                    help="initial view; also the recorded view when there is no window")
    a = ap.parse_args()

    from pi_decolor import Decolorizer
    from decolorizers import get_any
    ours = Decolorizer(a.weights)
    bt709 = get_any("bt709")
    rival = get_any(a.rival) if a.rival else None
    print(f"loaded selector: {len(ours.protos)} prototypes")
    if rival:
        print(f"rival method in the comparison: {a.rival}")

    score = {}                      # last computed metric per pane, held between refreshes
    if a.metrics:
        from metrics import contrast_metrics
        print(f"live E-score on (every {a.metric_every} frames) -- costs frame rate")

    def e_of(key, rgb_img, gray):
        """E-score, recomputed only every --metric-every frames."""
        if not a.metrics:
            return ""
        if n % a.metric_every == 0 or key not in score:
            score[key] = contrast_metrics(rgb_img, gray, n_pairs=2000)["E"]
        return f"   E={score[key]:.3f}"

    cam = open_camera(a.cam, (a.width, int(a.width * 3 / 4)), a.backend)
    if getattr(cam, "backend", None) is not None:
        print(f"camera opened with OpenCV backend {cam.backend}")

    headless = a.headless
    if not headless and not gui_available():
        headless = True
        a.record = True
        print("\n" + "=" * 72)
        print("This OpenCV build has no GUI, so there is no window to show. Capture works")
        print("fine -- only display is missing. Recording to mp4 instead so the run still")
        print("produces something you can watch. Use --snap-every to also save stills.")
        print("")
        print("To get the interactive demo back:")
        print("    pip uninstall -y opencv-python-headless opencv-python")
        print("    pip install opencv-python")
        print("Check afterwards with:  pip list | findstr opencv   (Windows)")
        print("Only ONE of the two may be installed -- they provide the same `cv2` module,")
        print("so installing pyiqa again will silently put the headless build back.")
        print("=" * 72 + "\n")

    mode, ema, rows, n = a.mode, {}, [], 0
    writer = None
    t_start = time.perf_counter()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            # Preserve the camera's aspect ratio. Squashing 640x480 into a square
            # distorts faces and makes the input pane look wrong next to a photo.
            fh, fw = frame.shape[:2]
            pw = a.proc
            ph = max(8, int(round(a.proc * fh / fw)))
            dw = a.disp
            dh = max(8, int(round(a.disp * fh / fw)))
            small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            panes, timings, clean = [], {}, {}   # clean: banner-free images for snapshots
            if mode in (1, 3):
                t0 = time.perf_counter()
                g = np.clip(bt709(rgb), 0, 1)
                dt = time.perf_counter() - t0
                ema["bt709"] = dt if "bt709" not in ema else 0.9 * ema["bt709"] + 0.1 * dt
                timings["bt709"] = dt
                disp = cv2.cvtColor(cv2.resize((g * 255).astype(np.uint8), (dw, dh)),
                                    cv2.COLOR_GRAY2BGR)
                clean["bt709"] = disp.copy()
                panes.append(banner(disp, "BT.709  (fixed standard)",
                                    f"{ema['bt709']*1e3:5.1f} ms   {1/ema['bt709']:5.1f} fps"
                                    + e_of("bt709", rgb, g)))
            if mode in (2, 3):
                t0 = time.perf_counter()
                g = np.clip(ours(rgb), 0, 1)
                dt = time.perf_counter() - t0
                ema["ours"] = dt if "ours" not in ema else 0.9 * ema["ours"] + 0.1 * dt
                timings["ours"] = dt
                disp = cv2.cvtColor(cv2.resize((g * 255).astype(np.uint8), (dw, dh)),
                                    cv2.COLOR_GRAY2BGR)
                clean["ours"] = disp.copy()
                wsel = ours.protos[ours.last_prototype]
                panes.append(banner(
                    disp, f"OURS  (prototype P{ours.last_prototype})",
                    f"{ema['ours']*1e3:5.1f} ms   {1/ema['ours']:5.1f} fps"
                    + e_of("ours", rgb, g),
                    f"w(R G B C)=({w4(wsel)})"))

            if rival is not None:
                t0 = time.perf_counter()
                gr = np.clip(rival(rgb), 0, 1)
                dt = time.perf_counter() - t0
                ema["rival"] = dt if "rival" not in ema else 0.9*ema["rival"] + 0.1*dt
                timings["rival"] = dt
                disp = cv2.cvtColor(cv2.resize((gr * 255).astype(np.uint8), (dw, dh)),
                                    cv2.COLOR_GRAY2BGR)
                clean[a.rival] = disp.copy()
                panes.append(banner(disp, f"{a.rival}  (published)",
                                    f"{ema['rival']*1e3:5.1f} ms   {1/ema['rival']:5.1f} fps"
                                    + e_of("rival", rgb, gr)))

            if mode == 4:
                t0 = time.perf_counter()
                grays, j = ours.all_prototypes(rgb)
                dt = time.perf_counter() - t0
                ema["grid"] = dt if "grid" not in ema else 0.9 * ema["grid"] + 0.1 * dt
                timings["ours"] = dt
                cell = a.grid_cell or max(72, a.disp // 2)
                cell_h = max(8, int(round(cell * fh / fw)))
                side = int(np.ceil(np.sqrt(len(grays))))
                tiles = []
                for i, g in enumerate(grays):
                    t = cv2.cvtColor(
                        cv2.resize((np.clip(g, 0, 1) * 255).astype(np.uint8), (cell, cell_h)),
                        cv2.COLOR_GRAY2BGR)
                    sel = (i == j)
                    cv2.rectangle(t, (0, 0), (cell - 1, cell_h - 1),
                                  (0, 215, 255) if sel else (60, 60, 60), 3 if sel else 1)
                    cv2.putText(t, f"P{i}" + ("  <" if sel else ""), (6, cell_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 215, 255) if sel else (200, 200, 200), 1, cv2.LINE_AA)
                    tiles.append(t)
                while len(tiles) % side:
                    tiles.append(np.zeros_like(tiles[0]))
                grid = np.vstack([np.hstack(tiles[r * side:(r + 1) * side])
                                  for r in range(len(tiles) // side)])
                clean["grid"] = grid.copy()
                panes.append(banner(grid, f"ALL {len(grays)} PROTOTYPES   selected: P{j}",
                                    f"{ema['grid']*1e3:5.1f} ms   {1/ema['grid']:5.1f} fps"))

            # the input pane always matches the pane height, so mode 4's taller grid
            # composes with hstack exactly like the 3-pane view does
            if mode == 4 and rival is not None and n == 1:
                print("note: mode 4 also draws the rival pane, so the window is wide. "
                      "Press 3 for the compact three-pane view, or drop --rival.")
            # Panes can differ in height -- the mode-4 grid is 4 tiles tall while a
            # BT.709 or rival pane is one. hstack requires equal heights, so scale
            # every pane up to the tallest before joining.
            if panes:
                oh = max(p_.shape[0] for p_ in panes)
                panes = [p_ if p_.shape[0] == oh else
                         cv2.resize(p_, (max(8, int(round(p_.shape[1] * oh / p_.shape[0]))), oh))
                         for p_ in panes]
            else:
                oh = dh
            ow = max(8, int(round(oh * fw / fh)))
            orig_clean = cv2.resize(frame, (ow, oh))
            clean["input"] = orig_clean.copy()
            orig = banner(orig_clean, f"Input RGB   (model sees {pw}x{ph})")
            view = np.hstack([orig] + panes)
            n += 1
            rows.append([n, f"{time.perf_counter()-t_start:.4f}"] +
                        [f"{timings.get(k, float('nan')):.5f}"
                         for k in ("bt709", "ours", "rival")])

            if a.record:
                # Do NOT hardcode the writer's frame rate. The loop runs at whatever the
                # slowest enabled pane allows -- with --rival that is ~9 fps, so a file
                # written at 20 fps plays back more than twice as fast as reality and
                # flatters the method. Measure the real rate over a short warm-up, then
                # open the writer with it, so the mp4 is correctly timed on its own and
                # needs no retiming pass afterwards.
                if writer is None and n >= 20:
                    true_fps = n / max(1e-6, time.perf_counter() - t_start)
                    writer = cv2.VideoWriter("demo_decolor.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, true_fps),
                                             (view.shape[1], view.shape[0]))
                    print(f"recording at the measured loop rate: {true_fps:.1f} fps")
                if writer is not None:
                    writer.write(view)

            # auto-snapshot every N frames (works with or without a window)
            if a.snap_every and n % a.snap_every == 0:
                save_snapshot(clean, view, a.snap_dir)

            if not headless:
                cv2.imshow("Decolorization — live on Raspberry Pi 5", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k in (ord("1"), ord("2"), ord("3"), ord("4")):
                    mode = k - ord("0")
                if k == ord("p"):
                    save_snapshot(clean, view, a.snap_dir)
            elif n % 20 == 0:
                print("  " + "  ".join(f"{k}:{1/v:5.1f}fps" for k, v in ema.items()))
            if a.frames and n >= a.frames:
                break
    finally:
        cam.stop()
        if writer is not None:
            writer.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass          # cleanup must never mask the exception that got us here
        if a.record:
            with open("demo_fps_log.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "wall_clock_s", "bt709_s", "ours_s", "rival_s"])
                w.writerows(rows)
            print("wrote demo_decolor.mp4 and demo_fps_log.csv")

    el = time.perf_counter() - t_start
    print(f"\n{n} frames in {el:.1f}s -> {n/max(el,1e-6):.1f} fps end-to-end "
          f"(this is the honest number: it includes every enabled pane)")
    print(f"per-method mean latency:")
    for k, v in ema.items():
        print(f"  {k:8s} {v*1e3:6.1f} ms   {1/v:5.1f} fps")


if __name__ == "__main__":
    main()
