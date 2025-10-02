import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import os
import numpy as np
import sklearn

MODEL_FILE = "password_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
FREQ_FILE = "freq_full.pkl"

# ---------------------------
# Load trained model
# ---------------------------
def load_model():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    with open(FREQ_FILE, "rb") as f:
        freq = pickle.load(f)
    return model, vectorizer, freq

model, vectorizer, freq = load_model()

# ---------------------------
# Password check function
# ---------------------------
def check_password():
    pwd = entry.get().strip()
    if not pwd:
        messagebox.showwarning("Warning", "Please enter a password!")
        return

    if pwd in freq:
        count = freq[pwd]
        total = sum(freq.values())
        norm_freq = count / total
        score = max(0.0, 1 - norm_freq * 1e5)
        result.set(
            f"⚠️ '{pwd}' is common!\n\n"
            f"Frequency: {count}\n"
            f"Normalized Frequency: {norm_freq:.6f}\n"
            f"Uniqueness Score: {score:.4f}"
        )
    else:
        vec = vectorizer.transform([pwd])
        pred = model.predict_proba(vec)[0][1]
        score = max(0.0, 1 - pred * 10)
        result.set(
            f"✅ '{pwd}' not found in dataset.\n\n"
            f"Predicted Commonness: {pred:.6f}\n"
            f"Uniqueness Score: {score:.4f}"
        )

# ---------------------------
# GUI Setup
# ---------------------------
root = tk.Tk()
root.title("🔐 Password Strength Checker")
root.geometry("600x400")
root.configure(bg="#f4f6f9")

# Style config
style = ttk.Style()
style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                foreground="white",
                background="#4CAF50",
                padding=10,
                borderwidth=0)
style.map("TButton",
          background=[("active", "#45a049")])

style.configure("TEntry", padding=6)

# Title Label
title = tk.Label(root,
                 text="Password Strength Checker",
                 font=("Segoe UI", 20, "bold"),
                 fg="#333",
                 bg="#f4f6f9")
title.pack(pady=20)

# Input frame
frame = tk.Frame(root, bg="#f4f6f9")
frame.pack(pady=10)

entry = ttk.Entry(frame, width=35, font=("Segoe UI", 14))
entry.grid(row=0, column=0, padx=10)

check_btn = ttk.Button(frame, text="Check Password", command=check_password)
check_btn.grid(row=0, column=1, padx=10)

# Result Label
result = tk.StringVar()
result_label = tk.Label(root,
                        textvariable=result,
                        wraplength=500,
                        justify="left",
                        font=("Segoe UI", 12),
                        bg="#f4f6f9",
                        fg="#444")
result_label.pack(pady=30)

root.mainloop()
