# ============================================================
#  Fake News Detection System - GUI (Hybrid: Transformer + Fact Checker)
# ============================================================

import tkinter as tk
from tkinter import messagebox, scrolledtext
import requests
import threading
import webbrowser

API_URL = "http://127.0.0.1:8000/predict"  # Ensure backend (FastAPI) is running

# ------------------------------------------------------------
# Function to open URLs in browser
# ------------------------------------------------------------
def open_url(url):
    try:
        webbrowser.open_new_tab(url)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open link: {e}")

# ------------------------------------------------------------
# Function to send text to API
# ------------------------------------------------------------
def analyze_news():
    news_text = news_input.get("1.0", tk.END).strip()
    if not news_text:
        messagebox.showwarning("Warning", "Please enter a news text.")
        return

    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, "🔍 Analyzing with AI models... Please wait.\n")

    def fetch_results():
        try:
            response = requests.post(API_URL, json={"text": news_text})
            if response.status_code == 200:
                data = response.json()
                display_results(data)
            else:
                result_box.delete("1.0", tk.END)
                result_box.insert(tk.END, f"❌ Error: {response.status_code}\n{response.text}")
        except Exception as e:
            result_box.delete("1.0", tk.END)
            result_box.insert(tk.END, f"⚠️ Unable to connect to backend:\n{e}")

    threading.Thread(target=fetch_results).start()

# ------------------------------------------------------------
# Function to display results from API
# ------------------------------------------------------------
def display_results(data):
    result_box.delete("1.0", tk.END)

    # ----- Input News -----
    result_box.insert(tk.END, f"📰 INPUT NEWS:\n{data['input_text']}\n\n")
    # Show detected language
    lang = data.get("language", {})
    lang_code = lang.get("code", "unknown")
    lang_name = lang.get("name", "Unknown")

    result_box.insert(
        tk.END,
        f"🌐 Detected Language: {lang_name} ({lang_code})\n\n"
    )

    # ----- Baseline Models -----
    result_box.insert(tk.END, "📊 BASELINE MODEL PREDICTIONS:\n")
    for model, vals in data["baselines"].items():
        result_box.insert(
            tk.END, f"  {model}: {vals['prediction']} (Confidence: {vals['confidence']}%)\n"
        )

    # ----- Transformer -----
    trans = data["transformer"]
    result_box.insert(
        tk.END,
        f"\n🤖 Transformer (XLM-RoBERTa): {trans['prediction']} (Confidence: {trans['confidence']}%)\n"
    )

    # ----- Hybrid Final Verdict -----
    result_box.insert(
        tk.END,
        f"\n🧠 Final Verdict (Hybrid: Transformer + Fact Checker): {data['final_verdict']}\n"
    )
    result_box.insert(tk.END, f"💬 Explanation: {data['reason']}\n\n")

    # -------- FACT CHECK RESULTS (from API) --------
    fact = data.get("fact_check", {})

    verdict = fact.get("verdict", "UNVERIFIED")
    confidence = fact.get("confidence", "Low")
    matches = fact.get("matches", [])

    result_box.insert(tk.END, "\n🌐 FACT CHECK RESULTS:\n")
    result_box.insert(tk.END, f"   Verdict: {verdict}  (Confidence: {confidence})\n")

    if isinstance(matches, list) and matches:
        result_box.insert(tk.END, "   Supporting headlines from trusted news sources:\n")
        for m in matches[:3]:  # show top 3
            source = m.get("source", "Unknown")
            title = m.get("title", "(no title)")
            sim = m.get("similarity", 0.0)
            result_box.insert(
                tk.END,
                f"      • [{source}] {title}  (Similarity: {sim:.2f})\n"
            )
    else:
        result_box.insert(
            tk.END,
            "   ⚠ No strong matching news articles found online for this statement.\n"
        )
    
# ------------------------------------------------------------
# Tkinter UI Setup
# ------------------------------------------------------------
root = tk.Tk()
root.title("Fake News Detection System (Hybrid AI + Fact Checker)")
root.geometry("960x720")
root.config(bg="#1e1e2f")

# Title
tk.Label(
    root,
    text="🧠 Fake News Detection System",
    font=("Arial", 22, "bold"),
    bg="#1e1e2f",
    fg="#00b0ff",
).pack(pady=15)

# Input
tk.Label(
    root,
    text="Enter News Text Below:",
    font=("Arial", 13),
    bg="#1e1e2f",
    fg="white",
).pack()

news_input = scrolledtext.ScrolledText(
    root, height=6, wrap="word", font=("Consolas", 12), bg="#2d2d44", fg="white"
)
news_input.pack(padx=20, pady=10, fill="x")

# Analyze Button
tk.Button(
    root,
    text="🔍 Analyze",
    font=("Arial", 13, "bold"),
    bg="#00b0ff",
    fg="white",
    relief="raised",
    command=analyze_news,
).pack(pady=10)

# Results Box
result_box = scrolledtext.ScrolledText(
    root, height=20, wrap="word", font=("Consolas", 11), bg="#2d2d44", fg="#e5e5e5"
)
result_box.pack(padx=20, pady=10, fill="both", expand=True)

# Footer
tk.Label(
    root,
    text="Developed for AIPS Project | © 2025",
    font=("Arial", 9),
    bg="#1e1e2f",
    fg="#aaaaaa",
).pack(side="bottom", pady=5)

root.mainloop()
