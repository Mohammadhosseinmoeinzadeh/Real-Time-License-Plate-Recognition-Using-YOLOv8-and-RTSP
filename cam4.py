import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import time
import av
import cv2
import numpy as np
from ultralytics import YOLO
import json
import queue


CONFIG_PATH = "config.json"

class PlateApp:
    def __init__(self, root, model_path):
        self.root = root
        self.model = YOLO(model_path)
        self.running = False
        self.save_path = ""
        self.last_plate_img = None
        self.frame_queue = queue.Queue(maxsize=1)

        self.root.title("تشخیص پلاک با PyAV - RTSP H.265")
        self.root.geometry("1000x800")

        self.load_config()

        self.rtsp_url_var = tk.StringVar()

        # فریم برای ورودی آدرس RTSP
        rtsp_frame = tk.Frame(root)
        rtsp_frame.pack(pady=(10, 0))

        rtsp_label = tk.Label(rtsp_frame, text="آدرس RTSP:", font=("Arial", 12, "bold"))
        rtsp_label.pack(side="left")

        self.rtsp_entry = tk.Entry(rtsp_frame, textvariable=self.rtsp_url_var, width=80)
        self.rtsp_entry.insert(0, "لطفا آدرس دوربین خود را وارد کنید")
        self.rtsp_entry.pack(side="left", padx=5)

        # 🔳 فریم اصلی شامل تصویر زنده و تصویر آخرین پلاک در کنار هم
        video_section = tk.Frame(root)
        video_section.pack(pady=10)

        # فریم نمایش دوربین
        camera_frame = tk.LabelFrame(video_section, text="تصویر زنده دوربین", padx=5, pady=5, bg="white")
        camera_frame.pack(side="left", padx=10)
        self.label_live = tk.Label(camera_frame, bg="black")
        self.label_live.pack()
        dummy_cam = Image.new("RGB", (640, 480), color=(0, 0, 0))
        cam_placeholder = ImageTk.PhotoImage(dummy_cam)
        self.label_live.configure(image=cam_placeholder)
        self.label_live.imgtk = cam_placeholder

        # فریم نمایش پلاک آخر
        plate_frame = tk.LabelFrame(video_section, text="آخرین پلاک ذخیره‌شده", padx=5, pady=5, bg="white")
        plate_frame.pack(side="left", padx=10)
        self.plate_preview = tk.Label(plate_frame, bg="black")
        self.plate_preview.pack()
        dummy_plate = Image.new("RGB", (320, 240), color=(0, 0, 0))
        plate_placeholder = ImageTk.PhotoImage(dummy_plate)
        self.plate_preview.configure(image=plate_placeholder)
        self.plate_preview.imgtk = plate_placeholder

        # دکمه‌ها افقی
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)

        start_img = Image.open("start_icon.png").resize((40, 40))
        start_icon = ImageTk.PhotoImage(start_img)
        self.start_btn = tk.Button(btn_frame, image=start_icon, text="شروع", compound="top", command=self.start, width=80, height=80)
        self.start_btn.image = start_icon
        self.start_btn.pack(side="left", padx=20)

        stop_img = Image.open("stop_icon.png").resize((40, 40))
        stop_icon = ImageTk.PhotoImage(stop_img)
        self.stop_btn = tk.Button(btn_frame, image=stop_icon, text="توقف", compound="top", command=self.stop, width=80, height=80)
        self.stop_btn.image = stop_icon
        self.stop_btn.pack(side="left", padx=20)

        path_img = Image.open("folder_icon.png").resize((40, 40))
        path_icon = ImageTk.PhotoImage(path_img)
        self.path_btn = tk.Button(btn_frame, image=path_icon, text="انتخاب مسیر ذخیره", compound="top", command=self.select_path, width=120, height=80)
        self.path_btn.image = path_icon
        self.path_btn.pack(side="left", padx=20)

        if not self.save_path:
            messagebox.showinfo("نیاز به انتخاب مسیر ذخیره‌سازی", "لطفاً مسیر ذخیره را انتخاب کنید.")
            self.select_path()

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                self.save_path = config.get("save_path", "")

    def save_config(self):
        with open(CONFIG_PATH, 'w') as f:
            json.dump({"save_path": self.save_path}, f)

    def select_path(self):
        path = filedialog.askdirectory()
        if path:
            self.save_path = path
            self.save_config()

    def start(self):
        if not self.save_path:
            messagebox.showwarning("خطا", "لطفاً ابتدا مسیر ذخیره را انتخاب کنید.")
            return
        if not self.running:
            self.running = True
            threading.Thread(target=self.process_stream, daemon=True).start()
            threading.Thread(target=self.analyze_frames, daemon=True).start()

    def stop(self):
        self.running = False

    def process_stream(self):
        rtsp_url = self.rtsp_url_var.get()
        if not rtsp_url or rtsp_url.startswith("لطفا"):
            messagebox.showerror("خطا", "لطفاً آدرس صحیح دوربین RTSP را وارد کنید.")
            self.running = False
            return
        try:
            container = av.open(rtsp_url, options={"rtsp_transport": "tcp"}, timeout=3)
        except av.AVError as e:
            messagebox.showerror("اتصال ناموفق", f"❌ خطا در اتصال به دوربین:\n{e}")
            return

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for packet in container.demux(stream):
            if not self.running:
                break
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                if img is None or img.shape[0] == 0:
                    continue
                if not self.frame_queue.full():
                    self.frame_queue.put(img)

    def analyze_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                img = self.frame_queue.get()
                results = self.model(img, verbose=False)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_plate = img[y1:y2, x1:x2]
                        if cropped_plate.size > 0 and self.save_path:
                            annotated = img.copy()
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            half_width = annotated.shape[1] // 2
                            plate_resized = cv2.resize(cropped_plate, (half_width, 100))
                            padding = np.zeros((100, annotated.shape[1] - half_width, 3), dtype=np.uint8)
                            plate_bar = np.hstack((plate_resized, padding))

                            combined = np.vstack((annotated, plate_bar))
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.putText(combined, timestamp, (combined.shape[1] - text_size[0] - 10, combined.shape[0] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            filename = f"plate_{int(time.time())}.jpg"
                            filepath = os.path.join(self.save_path, filename)
                            cv2.imwrite(filepath, combined)

                            self.last_plate_img = combined.copy()
                            self.show_saved_plate(combined)
                            break

                self.show_frame(img)

    def show_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label_live.imgtk = imgtk
        self.label_live.configure(image=imgtk)

    def show_saved_plate(self, img):
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.plate_preview.imgtk = imgtk
        self.plate_preview.configure(image=imgtk)

if __name__ == "__main__":
    model_path = "<YOUR_MODEL_ADDRESS>"
    root = tk.Tk()
    app = PlateApp(root, model_path)
    root.mainloop()
