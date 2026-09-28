import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_TITLE = "X-Ray DIP Lab — Enhancement & Segmentation"
BG = "#0f172a"
PANEL = "#111c33"
PANEL_2 = "#172441"
TEXT = "#e5eefc"
MUTED = "#9db0cc"
ACCENT = "#55d6be"
ACCENT_2 = "#7aa7ff"
WARNING = "#f2c14e"


class XRayDIPApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(1000, 700)
        self.root.configure(bg=BG)

        self.original = None
        self.enhanced = None
        self.mask = None
        self.ground_truth = None
        self.display_refs = {}

        self._build_style()
        self._build_ui()
        self._set_status("Ready. Load a grayscale X-ray image to begin.")

    def _build_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TButton", background=PANEL_2, foreground=TEXT, padding=(10, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#24385f")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#062c2a", padding=(12, 8), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#76ead3")])
        style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2, foreground=TEXT)
        style.configure("TScale", background=PANEL)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.configure("Metric.TLabel", background=PANEL_2, foreground=TEXT, padding=8, font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=24, pady=(20, 10))
        ttk.Label(header, text="X-Ray DIP Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="A non-diagnostic educational pipeline for enhancement, segmentation, and mask evaluation", style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))

        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=24, pady=(0, 12))
        ttk.Button(toolbar, text="Load X-ray", command=self.load_image, style="Accent.TButton").pack(side="left")
        ttk.Button(toolbar, text="Load ground-truth mask", command=self.load_ground_truth).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Run full pipeline", command=self.run_pipeline).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_result).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="right")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        main.columnconfigure(0, weight=0, minsize=285)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main, style="Panel.TFrame", padding=16)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._build_controls(controls)

        viewer = ttk.Frame(main, style="Panel.TFrame", padding=14)
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)
        self._build_viewer(viewer)

    def _build_controls(self, parent):
        ttk.Label(parent, text="PIPELINE CONTROLS", style="Section.TLabel").pack(anchor="w", pady=(0, 14))

        ttk.Label(parent, text="Preprocessing / noise reduction").pack(anchor="w")
        self.denoise_var = tk.StringVar(value="Median")
        ttk.Combobox(parent, textvariable=self.denoise_var, values=["None", "Median", "Gaussian"], state="readonly").pack(fill="x", pady=(4, 12))

        ttk.Label(parent, text="Enhancement").pack(anchor="w")
        self.enhance_var = tk.StringVar(value="CLAHE")
        ttk.Combobox(parent, textvariable=self.enhance_var, values=["CLAHE", "Contrast stretch", "Gamma correction", "Unsharp mask"], state="readonly").pack(fill="x", pady=(4, 12))

        ttk.Label(parent, text="Segmentation").pack(anchor="w")
        self.segment_var = tk.StringVar(value="Otsu threshold")
        ttk.Combobox(parent, textvariable=self.segment_var, values=["Otsu threshold", "Adaptive threshold", "Canny edges"], state="readonly").pack(fill="x", pady=(4, 12))

        ttk.Label(parent, text="Post-processing").pack(anchor="w")
        self.morph_var = tk.StringVar(value="Opening + closing")
        ttk.Combobox(parent, textvariable=self.morph_var, values=["None", "Opening", "Closing", "Opening + closing"], state="readonly").pack(fill="x", pady=(4, 12))

        ttk.Label(parent, text="Morphology kernel size").pack(anchor="w")
        self.kernel_var = tk.IntVar(value=5)
        tk.Scale(parent, from_=1, to=15, orient="horizontal", variable=self.kernel_var, resolution=2, bg=PANEL, fg=TEXT, troughcolor=PANEL_2, highlightthickness=0).pack(fill="x", pady=(2, 12))

        ttk.Label(parent, text="Gamma value").pack(anchor="w")
        self.gamma_var = tk.DoubleVar(value=1.0)
        tk.Scale(parent, from_=0.3, to=2.5, orient="horizontal", variable=self.gamma_var, resolution=0.1, bg=PANEL, fg=TEXT, troughcolor=PANEL_2, highlightthickness=0).pack(fill="x", pady=(2, 12))

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Show segmentation overlay", variable=self.overlay_var, command=self.refresh_view).pack(anchor="w", pady=(2, 14))

        ttk.Label(parent, text="EVALUATION", style="Section.TLabel").pack(anchor="w", pady=(8, 10))
        ttk.Label(parent, text="Load a binary reference mask to calculate Dice and IoU.", wraplength=245, foreground=MUTED).pack(anchor="w")
        self.dice_label = ttk.Label(parent, text="Dice: —", style="Metric.TLabel")
        self.dice_label.pack(fill="x", pady=(12, 5))
        self.iou_label = ttk.Label(parent, text="IoU: —", style="Metric.TLabel")
        self.iou_label.pack(fill="x")

        ttk.Label(parent, text="Educational use only — not for diagnosis.", foreground=WARNING, wraplength=245).pack(side="bottom", anchor="w", pady=(20, 0))

    def _build_viewer(self, parent):
        self.view_title = ttk.Label(parent, text="No image loaded", style="Section.TLabel")
        self.view_title.grid(row=0, column=0, sticky="w", pady=(0, 10))
        canvas_frame = tk.Frame(parent, bg="#0b1222")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="#0b1222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.refresh_view())
        self.status = ttk.Label(parent, text="", foreground=MUTED)
        self.status.grid(row=2, column=0, sticky="w", pady=(10, 0))

    def _set_status(self, text):
        self.status.config(text=text)

    def load_image(self):
        path = filedialog.askopenfilename(title="Select X-ray image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Load error", "The selected file could not be read as an image.")
            return
        self.original = image
        self.enhanced = None
        self.mask = None
        self.ground_truth = None
        self.dice_label.config(text="Dice: —")
        self.iou_label.config(text="IoU: —")
        self.view_title.config(text=f"Original: {Path(path).name}")
        self._set_status(f"Loaded {image.shape[1]} × {image.shape[0]} grayscale pixels.")
        self.refresh_view()

    def load_ground_truth(self):
        if self.original is None:
            messagebox.showinfo("Load an X-ray first", "Please load the original X-ray image before loading a reference mask.")
            return
        path = filedialog.askopenfilename(title="Select binary ground-truth mask", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            messagebox.showerror("Load error", "The selected mask could not be read.")
            return
        self.ground_truth = cv2.resize(mask, (self.original.shape[1], self.original.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.ground_truth = (self.ground_truth > 127).astype(np.uint8) * 255
        self._set_status(f"Reference mask loaded: {Path(path).name}. Run the pipeline to calculate metrics.")
        if self.mask is not None:
            self.update_metrics()

    def preprocess(self, image):
        mode = self.denoise_var.get()
        if mode == "Median":
            return cv2.medianBlur(image, 5)
        if mode == "Gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        return image.copy()

    def enhance(self, image):
        mode = self.enhance_var.get()
        if mode == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        if mode == "Contrast stretch":
            low, high = np.percentile(image, (2, 98))
            if high <= low:
                return image.copy()
            stretched = (image.astype(np.float32) - low) * 255.0 / (high - low)
            return np.clip(stretched, 0, 255).astype(np.uint8)
        if mode == "Gamma correction":
            gamma = max(0.1, float(self.gamma_var.get()))
            lut = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)], dtype=np.uint8)
            return cv2.LUT(image, lut)
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)

    def segment(self, image):
        mode = self.segment_var.get()
        if mode == "Otsu threshold":
            _threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask
        if mode == "Adaptive threshold":
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4)
        return cv2.Canny(image, 60, 150)

    def postprocess(self, mask):
        mode = self.morph_var.get()
        if mode == "None":
            return mask
        size = max(1, int(self.kernel_var.get()))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if mode == "Opening":
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if mode == "Closing":
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def run_pipeline(self):
        if self.original is None:
            messagebox.showinfo("Load an image first", "Please load an X-ray image before running the pipeline.")
            return
        preprocessed = self.preprocess(self.original)
        self.enhanced = self.enhance(preprocessed)
        self.mask = self.postprocess(self.segment(self.enhanced))
        self.view_title.config(text="Pipeline output — enhanced image with segmentation")
        self.update_metrics()
        self.refresh_view()
        self._set_status("Pipeline complete: preprocessing → enhancement → segmentation → morphology.")

    def update_metrics(self):
        if self.mask is None or self.ground_truth is None:
            return
        prediction = self.mask > 0
        truth = self.ground_truth > 0
        intersection = np.logical_and(prediction, truth).sum()
        dice_den = prediction.sum() + truth.sum()
        union = np.logical_or(prediction, truth).sum()
        dice = (2.0 * intersection / dice_den) if dice_den else 1.0
        iou = (intersection / union) if union else 1.0
        self.dice_label.config(text=f"Dice: {dice:.4f}")
        self.iou_label.config(text=f"IoU: {iou:.4f}")

    def refresh_view(self):
        if self.original is None or not hasattr(self, "canvas"):
            return
        if self.enhanced is not None and self.mask is not None:
            display = cv2.cvtColor(self.enhanced, cv2.COLOR_GRAY2RGB)
            if self.overlay_var.get():
                overlay = np.zeros_like(display)
                overlay[:, :, 1] = 220
                alpha = (self.mask > 0).astype(np.float32)[:, :, None] * 0.38
                display = (display * (1 - alpha) + overlay * alpha).astype(np.uint8)
            label = "Enhanced + mask overlay" if self.overlay_var.get() else "Enhanced image"
        else:
            display = cv2.cvtColor(self.original, cv2.COLOR_GRAY2RGB)
            label = "Original image"
        height, width = display.shape[:2]
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        scale = min(canvas_w / width, canvas_h / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(display, new_size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=image)
        self.display_refs["main"] = photo
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
        self.canvas.create_text(14, 14, text=label, fill=ACCENT, anchor="nw", font=("Segoe UI", 10, "bold"))

    def save_result(self):
        if self.original is None:
            messagebox.showinfo("Nothing to save", "Load and process an X-ray first.")
            return
        output_dir = filedialog.askdirectory(title="Choose output folder")
        if not output_dir:
            return
        base = Path(output_dir)
        cv2.imwrite(str(base / "original.png"), self.original)
        if self.enhanced is not None:
            cv2.imwrite(str(base / "enhanced.png"), self.enhanced)
        if self.mask is not None:
            cv2.imwrite(str(base / "segmentation_mask.png"), self.mask)
            overlay = cv2.cvtColor(self.enhanced if self.enhanced is not None else self.original, cv2.COLOR_GRAY2BGR)
            overlay[self.mask > 0] = (80, 210, 150)
            cv2.imwrite(str(base / "overlay.png"), cv2.addWeighted(cv2.cvtColor(self.enhanced if self.enhanced is not None else self.original, cv2.COLOR_GRAY2BGR), 0.68, overlay, 0.32, 0))
        messagebox.showinfo("Saved", f"Results saved to:\n{base}")

    def reset(self):
        self.original = None
        self.enhanced = None
        self.mask = None
        self.ground_truth = None
        self.view_title.config(text="No image loaded")
        self.dice_label.config(text="Dice: —")
        self.iou_label.config(text="IoU: —")
        self.canvas.delete("all")
        self._set_status("Ready. Load a grayscale X-ray image to begin.")


def main():
    root = tk.Tk()
    XRayDIPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

