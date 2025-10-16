# Lightweight Tkinter UI for demo
# Exercises: Pull-ups, Squats, Bench Press, Bicep Curls.
# Swap out get_prediction_from_model() with ML later

import random
import tkinter as tk
from tkinter import ttk
from typing import Dict, Tuple

EXERCISES = ["Pull-up", "Squat", "Bench Press", "Bicep Curl"]
UPDATE_MS = 333  # ~3 Hz
reps = 0 

# Replace with real model
##########################################################################################
_prev = None
def get_prediction_from_model() -> Tuple[str, float, Dict[str, float]]:
    """Replace with real model.
    Must return: (label:str, confidence:float 0..1, scores:Dict[str,float])."""
    global _prev
    base = [0.25]*4
    if _prev in EXERCISES:
        i = EXERCISES.index(_prev); base = [0.2]*4; base[i] = 0.4
    raw = [max(0.001, b + random.uniform(-0.05, 0.05)) for b in base]
    s = sum(raw); probs = [r/s for r in raw]
    scores = dict(zip(EXERCISES, probs))
    label = EXERCISES[max(range(4), key=lambda i: probs[i])]
    _prev = label
    return label, scores[label], scores
##########################################################################################

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
