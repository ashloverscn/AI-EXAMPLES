import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import sounddevice as sd
import soundfile as sf

# Configuration
SAMPLE_RATE = 16000  # 16kHz is ideal for speech/audio deep learning models
CHANNELS = 1

class AudioCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Learning Audio Data Collector")
        self.root.geometry("400x450")
        self.root.resizable(False, False)

        self.q = queue.Queue()
        self.is_recording = False
        self.recorded_frames = []
        self.last_saved_file = None

        # --- UI ELEMENTS ---
        
        # Title Label
        title_label = tk.Label(root, text="Audio Dataset Recorder", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Speaker / Class Name
        tk.Label(root, text="Enter Class / Speaker Name:", font=("Arial", 10)).pack(anchor="w", padx=30)
        self.class_entry = ttk.Entry(root, width=30, font=("Arial", 11))
        self.class_entry.pack(pady=5, padx=30, anchor="w")
        self.class_entry.insert(0, "person_1")

        # Duration Entry
        tk.Label(root, text="Recording Duration (seconds):", font=("Arial", 10)).pack(anchor="w", padx=30, pady=(10, 0))
        self.duration_entry = ttk.Entry(root, width=10, font=("Arial", 11))
        self.duration_entry.pack(pady=5, padx=30, anchor="w")
        self.duration_entry.insert(0, "3")

        # Status Display Box
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = tk.Label(root, textvariable=self.status_var, font=("Arial", 11, "italic"), fg="blue")
        status_label.pack(pady=20)

        # Buttons
        self.rec_button = tk.Button(root, text="🔴 Record Audio", bg="#ff4d4d", fg="white", font=("Arial", 11, "bold"), width=25, command=self.start_recording_thread)
        self.rec_button.pack(pady=5)

        self.play_button = tk.Button(root, text="▶ Play Last Recording", bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=25, command=self.play_audio)
        self.play_button.pack(pady=5)

        # Instructions Footer
        instructions = "Files will be saved automatically inside a\n'dataset/<speaker_name>' folder."
        tk.Label(root, text=instructions, fg="gray", font=("Arial", 9)).pack(pady=15)

    def callback(self, indata, frames, time_info, status):
        """This is called for every audio block by sounddevice."""
        if status:
            print(status)
        self.q.put(indata.copy())

    def start_recording_thread(self):
        """Runs the recording logic in a separate thread so GUI doesn't freeze."""
        if self.is_recording:
            return
        
        class_name = self.class_entry.get().strip()
        if not class_name:
            messagebox.showerror("Error", "Please enter a valid class/speaker name!")
            return

        try:
            duration = float(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Duration must be a valid number!")
            return

        # Disable button and start thread
        self.is_recording = True
        self.rec_button.config(state=tk.DISABLED, bg="gray")
        
        threading.Thread(target=self.record_audio_process, args=(class_name, duration)).start()

    def record_audio_process(self, class_name, duration):
        self.status_var.set("Status: Recording... Speak now!")
        self.recorded_frames = []

        # Create target directory dataset/class_name
        save_dir = os.path.join("dataset", class_name)
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename using timestamp/count
        existing_files = len(os.listdir(save_dir))
        filename = os.path.join(save_dir, f"sample_{existing_files + 1}.wav")
        self.last_saved_file = filename

        try:
            # Start sounddevice input stream
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback):
                # Simple loop to collect chunks for the set duration
                for _ in range(int(duration * 10)):
                    time.sleep(0.1)
                    
            # Gather data from queue
            while not self.q.empty():
                self.recorded_frames.append(self.q.get())

            # Convert to numpy array and save via soundfile
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            sf.write(filename, audio_data, SAMPLE_RATE)

            self.status_var.set(f"Status: Saved to {filename}")
            messagebox.showinfo("Success", f"Audio successfully saved as:\n{filename}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Status: Recording failed.")

        finally:
            self.is_recording = False
            self.rec_button.config(state=tk.NORMAL, bg="#ff4d4d")

    def play_audio(self):
        """Plays back the last recorded file through the default audio output device."""
        if not self.last_saved_file or not os.path.exists(self.last_saved_file):
            messagebox.showwarning("Warning", "No recent recording found to play!")
            return

        def playback():
            self.status_var.set(f"Status: Playing {os.path.basename(self.last_saved_file)}...")
            data, fs = sf.read(self.last_saved_file)
            sd.play(data, fs)
            sd.wait()
            self.status_var.set("Status: Playback finished.")

        threading.Thread(target=playback).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioCollectorApp(root)
    root.mainloop()