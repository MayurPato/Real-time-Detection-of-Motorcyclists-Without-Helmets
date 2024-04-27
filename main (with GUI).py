import cv2
import tkinter as tk
from tkinter import filedialog, ttk
import torch
import numpy as np
import pandas as pd
import concurrent.futures

path = ".../custom_weights.pt"
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload = True)
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)

my_file = open(".../coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

def run_model1(frame):
    results1 = model1(frame)
    px1 = pd.DataFrame(results1.xyxy[0])
    list1 = []
    for index, row in px1.iterrows():
        x1, y1, x2, y2 = map(int, row[[0, 1, 2, 3]])
        d = int(row[5])
        if d == 2:
            list1.append([x1, y1, x2, y2])
    return list1

def run_model2(frame):
    results2 = model2(frame)
    px2 = pd.DataFrame(results2.xyxy[0])
    list2 = []
    for index, row in px2.iterrows():
        x1, y1, x2, y2 = map(int, row[[0, 1, 2, 3]])
        c = class_list[int(row[5])]
        if 'motorcycle' in c:
            list2.append([x1, y1 - int(0.2 * frame.shape[0]), x2, y2])
    return list2

def find_point_box_relationship(frame, list1, list2):
    for box1 in list1:
        for box2 in list2:
            x1, y1, x2, y2 = box1
            p_inside = box2[0] <= x1 <= box2[2] and box2[1] <= y1 <= box2[3] and box2[0] <= x2 <= box2[2] and box2[1] <= y2 <= box2[3]
            if p_inside:
                cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
                cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Head", (box1[0], box1[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

def select_file():
    file_path = filedialog.askopenfilename(title="Select a Video File",
                                           filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        process_video(file_path)

def process_video(file_path):
    skip_frames = int(skip_spinbox.get())
    
    cap = cv2.VideoCapture(file_path)
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            count += 1
            if count % skip_frames != 0:
                continue
            
            if not use_original_resolution.get():
                resize_width = int(width_spinbox.get())
                resize_height = int(height_spinbox.get())
                frame = cv2.resize(frame, (resize_width, resize_height))
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            global image_height, image_width
            image_height, image_width, _ = frame.shape
            
            future1 = executor.submit(run_model1, frame)
            future2 = executor.submit(run_model2, frame)
            
            list1 = future1.result()
            list2 = future2.result()
            
            if list1 and list2:
                find_point_box_relationship(frame, list1, list2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("System Output", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("SafetyVision")
root.geometry("600x420")

skip_spinbox = None
width_spinbox = None
height_spinbox = None
use_original_resolution = None

style = ttk.Style(root)
style.configure("TLabel", font=("Arial", 14))
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TFrame", relief=tk.SUNKEN, borderwidth=2)

welcome_label = tk.Label(root, text="Welcome to SafetyVision's Motorcycle Helmet Compliance System!", font=("Arial", 16, "bold"))
welcome_label.pack(pady=20)

description_text = tk.Label(root, text="""The Motorcycle Helmet Compliance System is specifically designed for road environments, leveraging live CCTV feeds to detect motorcycle riders not wearing helmets.""", font=("Arial", 12), wraplength=500)
description_text.pack(pady=5)

continue_button = tk.Button(root, text="Continue", font=("Arial", 14), command=lambda: [welcome_label.pack_forget(), description_text.pack_forget(), continue_button.pack_forget(), show_gui()])
continue_button.pack(pady=70)

def show_gui():
    global skip_spinbox, width_spinbox, height_spinbox, use_original_resolution

    frame_options = ttk.LabelFrame(root, text="Pre-Processing Options", padding=20)
    frame_options.pack(pady=40, padx=60, fill=tk.BOTH, expand=True)

    skip_label = ttk.Label(frame_options, text="Skip Frames:")
    skip_label.grid(row=0, column=0, pady=10, padx=10)
    skip_spinbox = ttk.Spinbox(frame_options, from_=1, to=10, width=5)
    skip_spinbox.grid(row=0, column=1, pady=10, padx=10)
    skip_spinbox.set(3)  

    use_original_resolution = tk.BooleanVar()
    original_resolution_check = ttk.Checkbutton(frame_options, text="Use Original Resolution", variable=use_original_resolution, onvalue=True, offvalue=False, command=toggle_resolution_options)
    original_resolution_check.grid(row=1, columnspan=2, pady=10, padx=10)

    width_label = ttk.Label(frame_options, text="Resize Frame Width:")
    width_label.grid(row=2, column=0, pady=10, padx=10)
    width_spinbox = ttk.Spinbox(frame_options, from_=100, to=2000, width=5, state=tk.NORMAL)
    width_spinbox.grid(row=2, column=1, pady=10, padx=10)
    width_spinbox.set(1020)  

    height_label = ttk.Label(frame_options, text="Resize Frame Height:")
    height_label.grid(row=3, column=0, pady=10, padx=10)
    height_spinbox = ttk.Spinbox(frame_options, from_=100, to=2000, width=5, state=tk.NORMAL)
    height_spinbox.grid(row=3, column=1, pady=10, padx=10)
    height_spinbox.set(500) 

    select_button = ttk.Button(root, text="Select a Video File", command=select_file)
    select_button.pack(pady=10)

    # exit_button = ttk.Button(root, text="Exit", command=root.quit)
    # exit_button.pack(pady=20)  

def toggle_resolution_options():
    if use_original_resolution.get():
        width_spinbox.config(state=tk.DISABLED)
        height_spinbox.config(state=tk.DISABLED)
    else:
        width_spinbox.config(state=tk.NORMAL)
        height_spinbox.config(state=tk.NORMAL)

root.mainloop()
