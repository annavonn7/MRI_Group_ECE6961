#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:44:48 2026

@author: ashlyn.nguyen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Import data
file_path = "US_MRI_GROUP.xlsx"

# Define variables
number_of_peaks = 7
minimum_peak_distance = 100

# Define columns 
data = pd.read_excel(file_path, sheet_name=0, header=None)
data = data.iloc[:, :2]
data.columns = ["time", "amplitude"]

# Convert values to numbers and remove invalid rows
data["time"] = pd.to_numeric(data["time"], errors="coerce")
data["amplitude"] = pd.to_numeric(data["amplitude"], errors="coerce")
data = data.dropna().sort_values("time").reset_index(drop=True)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.plot(data["time"], data["amplitude"], color="blue", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Amplitude vs. Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find candidate peaks
candidate_indices, properties = find_peaks(
    data["amplitude"].to_numpy(),
    prominence=0,
    distance=minimum_peak_distance
)

# Keep the seven peaks with the greatest prominence/peak
if len(candidate_indices) > number_of_peaks:
    strongest = np.argsort(properties["prominences"])[-number_of_peaks:]

    peak_indices = candidate_indices[strongest]
    peak_prominences = properties["prominences"][strongest]
else:
    peak_indices = candidate_indices
    peak_prominences = properties["prominences"]

# Put the selected peaks in chronological order
time_order = np.argsort(data.iloc[peak_indices]["time"].to_numpy())
peak_indices = peak_indices[time_order]
peak_prominences = peak_prominences[time_order]

# Create the output table
peaks = data.iloc[peak_indices].copy()
peaks["prominence"] = peak_prominences

# Label peaks: Initial bang, A, B, C, D, E, F
labels = ["Initial bang"]

for index in range(1, len(peaks)):
    labels.append(chr(64 + index))

peaks.insert(0, "peak", labels)

# Calculate time relative to the initial bang
if not peaks.empty:
    initial_time = peaks["time"].iloc[0]
    peaks["time_after_initial_bang"] = peaks["time"] - initial_time

peaks = peaks.rename(columns={"time": "arrival_time"})

peaks = peaks[
    [
        "peak",
        "arrival_time",
        "time_after_initial_bang",
        "amplitude",
        "prominence"
    ]
]

# Print and save the results
print(peaks.to_string(index=False))
peaks.to_excel("peak_arrival_times.xlsx", index=False)

# Plot the seven selected peaks
plt.figure(figsize=(12, 6))

plt.plot(
    data["time"],
    data["amplitude"],
    color="blue",
    linewidth=1,
    label="Signal"
)

plt.scatter(
    peaks["arrival_time"],
    peaks["amplitude"],
    color="red",
    s=60,
    zorder=3,
    label="Selected peaks"
)

for _, peak in peaks.iterrows():
    plt.annotate(
        peak["peak"],
        (peak["arrival_time"], peak["amplitude"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center"
    )

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Seven Most Prominent Peaks")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("selected_peaks.png", dpi=300)
plt.show()

print("\nResults saved as peak_arrival_times.xlsx")
print("Labeled plot saved as selected_peaks.png")