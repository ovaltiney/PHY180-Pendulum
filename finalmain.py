import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import os
import glob

# --- USER SETTINGS ---
video_dir = "videos"            # folder with input .mp4 files
scale_cm_per_px = 0.0265        # fixed calibration (cm per pixel)
pendulum_length_cm = 16.5       # string length in cm (hardcoded)
output_dir = "results_batch"
os.makedirs(output_dir, exist_ok=True)
summary_file = os.path.join(output_dir, "summary.txt")

# --- HSV range for pendulum bob (from tuner) ---
lower_color = np.array([94, 0, 109])
upper_color = np.array([179, 255, 255])

# --- Fitting function for exponential decay ---
def exp_decay(t, A, gamma, C):
    return A * np.exp(-gamma * t) + C

# --- Collect summary rows ---
summary_rows = []

# --- Loop through all .mp4 files in video_dir ---
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
if not video_files:
    raise RuntimeError("No .mp4 files found in the video directory.")

for video_path in video_files:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing {base_name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Skipping {base_name}, could not open file.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    time_vals, x_positions = [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t = frame_count / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            x_positions.append(cx)
            time_vals.append(t)

    cap.release()

    time_vals = np.array(time_vals)
    x_positions = np.array(x_positions)

    if len(x_positions) < 2:
        print(f"Not enough tracking data for {base_name}. Skipping.")
        continue

    # --- Convert to displacement in cm and radians ---
    x_eq = np.mean(x_positions)
    disp_cm = (x_positions - x_eq) * scale_cm_per_px
    disp_rad = np.arcsin(disp_cm / pendulum_length_cm)

    # --- Find peaks/troughs for amplitude ---
    peaks, _ = find_peaks(disp_rad, prominence=0.01)
    troughs, _ = find_peaks(-disp_rad, prominence=0.01)
    extrema = np.sort(np.concatenate([peaks, troughs]))
    amp_times = time_vals[extrema]
    amplitudes_cm = np.abs(disp_cm[extrema])  # amplitude in cm

    # --- Save amplitude table (in cm) ---
    amp_table = pd.DataFrame({"Time (s)": amp_times, "Horizontal Amplitude (cm)": amplitudes_cm})
    txt_file = os.path.join(output_dir, f"{base_name}_amplitude.txt")
    amp_table.to_csv(txt_file, sep="\t", index=False, float_format="%.6f")

    # --- Estimate period ---
    periods = []
    if len(peaks) > 1:
        periods.extend(np.diff(time_vals[peaks]))
    if len(troughs) > 1:
        periods.extend(np.diff(time_vals[troughs]))

    avg_period, std_period, n_osc = np.nan, np.nan, 0
    if len(periods) > 0:
        periods = np.array(periods)
        avg_period, std_period = np.mean(periods), np.std(periods)
        n_osc = len(periods)
        print(f"Period = {avg_period:.3f} ± {std_period:.3f} s, N_osc = {n_osc}")

    # --- Amplitude decay fit ---
    gamma, Q_factor, tau = np.nan, np.nan, np.nan
    plt.figure(figsize=(8,5))
    plt.plot(amp_times, amplitudes_cm, "ro", label="Amplitude data")

    try:
        p0 = [np.max(amplitudes_cm), 0.1, np.min(amplitudes_cm)]
        popt, _ = curve_fit(exp_decay, amp_times, amplitudes_cm, p0=p0)
        fit_times = np.linspace(amp_times[0], amp_times[-1], 200)
        plt.plot(fit_times, exp_decay(fit_times, *popt), "b-", label=f"Exp fit: γ={popt[1]:.3f}")
        gamma = popt[1]
        tau = 1.0 / gamma
        if not np.isnan(avg_period):
            Q_factor = np.pi / (gamma * avg_period)
            print(f"Damping γ = {gamma:.4f}, τ = {tau:.2f}, Q = {Q_factor:.2f}")
    except RuntimeError:
        print("Exponential fit failed.")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (cm)")
    plt.title("Amplitude vs Time (Decay)")
    plt.legend()
    plt.grid(True)
    amp_png = os.path.join(output_dir, f"{base_name}_amplitude_decay.png")
    plt.savefig(amp_png, dpi=300)
    plt.close()

    # --- Displacement vs Time plot with peaks marked ---
    plt.figure(figsize=(10,5))
    plt.plot(time_vals, disp_rad, label="Displacement (rad)")
    plt.scatter(time_vals[extrema], disp_rad[extrema], color="red", s=30, label="Peaks/Troughs")
    plt.axhline(0, color="k", linestyle="--", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (rad)")
    plt.title("Pendulum Displacement vs Time")
    plt.legend()
    plt.grid(True)
    disp_png = os.path.join(output_dir, f"{base_name}_displacement.png")
    plt.savefig(disp_png, dpi=300)
    plt.close()

    # --- Append to summary ---
    summary_rows.append({
        "Filename": base_name,
        "y": avg_period,
        "yerr": std_period,
        "N_osc": n_osc,
        "tau": tau,
        "Q": Q_factor
    })

# --- Save summary for all videos ---
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_file, sep="\t", index=False, float_format="%.6f")
print(f"\nSummary saved to {summary_file}")
