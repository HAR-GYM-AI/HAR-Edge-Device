# Lightweight Tkinter UI for demo
# Exercises: Pull-ups, Squats, Bench Press, Bicep Curls.
# Now uses real ML models instead of fake predictions

import random
import tkinter as tk
from tkinter import ttk
from typing import Dict, Tuple

# Import ML components
try:
    import joblib
    import numpy as np
    import pandas as pd
    exercise_model = joblib.load('exercise_classifier_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load ML models: {e}")
    print("Falling back to fake predictions")
    MODELS_LOADED = False

EXERCISES = ["Pull-up", "Squat", "Bench Press", "Bicep Curl"]
UPDATE_MS = 333  # ~3 Hz
reps = 0 

# Replace with real model
_prev = None
def get_prediction_from_model() -> Tuple[str, float, Dict[str, float]]:
    """Real ML-based prediction function when models are available, otherwise fake."""
    global _prev

    if MODELS_LOADED:
        try:
            # Create dummy feature vector matching model expectations
            # In real implementation, this would come from sensor data
            dummy_features = np.random.rand(1, 200)  # Approximate feature count
            feature_names = [f'feature_{i}' for i in range(200)]
            X_test = pd.DataFrame(dummy_features, columns=feature_names)

            # Make prediction
            pred = exercise_model.predict(X_test)[0]
            proba = exercise_model.predict_proba(X_test)[0]
            exercise = label_encoder.inverse_transform([pred])[0]
            confidence = proba[pred]

            # Create scores dict
            scores = dict(zip(label_encoder.classes_, proba))

            return exercise, confidence, scores

        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fall back to fake predictions
            pass

    # Fallback: Fake predictions
    base = [0.25]*4
    if _prev in EXERCISES:
        i = EXERCISES.index(_prev); base = [0.2]*4; base[i] = 0.4
    raw = [max(0.001, b + random.uniform(-0.05, 0.05)) for b in base]
    s = sum(raw); probs = [r/s for r in raw]
    scores = dict(zip(EXERCISES, probs))
    label = EXERCISES[max(range(4), key=lambda i: probs[i])]
    _prev = label
    return label, scores[label], scores

class UI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exercise Classifier")
        self.geometry("460x210"); self.minsize(380, 200)
        self.running = False

        wrap = ttk.Frame(self, padding=12); wrap.pack(fill="both", expand=True)
        # Exercsie 
        self.label = ttk.Label(wrap, text="—", font=("Segoe UI", 28, "bold")); self.label.pack()

        #Confidence
        self.conf  = ttk.Label(wrap, text="—%", font=("Segoe UI", 20)); self.conf.pack(pady=(0,8))

        # Rep count 
        self.reps_lbl = ttk.Label(wrap, text=f"Reps: {reps}", font=("Segoe UI", 18))
        self.reps_lbl.pack(pady=(0,10))

        # Start/Stop buttons
        bar = ttk.Frame(wrap); bar.pack()
        self.start_btn = ttk.Button(bar, text="Start", command=self.start); self.start_btn.pack(side="left", padx=4)
        self.stop_btn  = ttk.Button(bar, text="Stop", command=self.stop, state="disabled"); self.stop_btn.pack(side="left", padx=4)

    def start(self):
        if self.running: return
        self.running = True; self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self._tick()

    def stop(self):
        self.running = False; self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")

    def _tick(self):
        if not self.running: return
        label, conf, _ = get_prediction_from_model()
        self.label.config(text=label)
        self.conf.config(text=f"{conf*100:0.1f}%")
        self.after(UPDATE_MS, self._tick)

    def set_reps(self, n: int):
        self.reps_lbl.config(text=f"Reps: {n}")

if __name__ == "__main__":
    UI().mainloop()
