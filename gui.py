#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BOTH, BOTTOM, END, LEFT, RIGHT, X, Y, Button, Checkbutton, E, Entry, Frame, IntVar, Label, Listbox, N, S, Scrollbar, StringVar, Tk, filedialog, ttk, Text

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


APP_TITLE = "pdf2mm GUI"


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1000x720")
        # Theme and basic styling
        self.style = ttk.Style()
        themes = set(self.style.theme_names())
        if "vista" in themes:
            self.style.theme_use("vista")
        elif "clam" in themes:
            self.style.theme_use("clam")
        self.style.configure("TButton", padding=6)
        self.style.configure("TLabel", padding=2)
        self.style.configure("TEntry", padding=2)

        self.pdf_path = StringVar()
        self.out_dir = StringVar(value="")
        self.format = StringVar(value="jsonl")
        self.ocr = StringVar(value="auto")
        self.caption_provider = StringVar(value="none")
        self.embed_provider = StringVar(value="none")
        self.max_pages = StringVar(value="0")
        self.min_cap_len = StringVar(value="6")
        self.lang = StringVar(value="eng")
        self.config = StringVar(value="")
        self.proc = None

        self._build_controls()
        self._build_tabs()
        self._build_status()

        # Root should make notebook expand
        self.root.update_idletasks()

    def _build_controls(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Options")
        frm.pack(fill=X, padx=10, pady=6)

        Label(frm, text="PDF:").grid(row=0, column=0, sticky=E)
        Entry(frm, textvariable=self.pdf_path, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self.browse_pdf).grid(row=0, column=2)

        Label(frm, text="Out dir:").grid(row=1, column=0, sticky=E)
        Entry(frm, textvariable=self.out_dir, width=60).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self.browse_out).grid(row=1, column=2)

        Label(frm, text="Format:").grid(row=2, column=0, sticky=E)
        ttk.Combobox(frm, textvariable=self.format, values=["jsonl", "yaml", "markdown"], width=15).grid(row=2, column=1, sticky="w", padx=4)

        Label(frm, text="OCR:").grid(row=2, column=2, sticky=E)
        ttk.Combobox(frm, textvariable=self.ocr, values=["auto", "yes", "no"], width=10).grid(row=2, column=3, sticky="w", padx=4)

        Label(frm, text="Caption:").grid(row=3, column=0, sticky=E)
        ttk.Combobox(frm, textvariable=self.caption_provider, values=["none", "openai", "blip"], width=15).grid(row=3, column=1, sticky="w", padx=4)

        Label(frm, text="Embed:").grid(row=3, column=2, sticky=E)
        ttk.Combobox(frm, textvariable=self.embed_provider, values=["none", "openai", "hf"], width=10).grid(row=3, column=3, sticky="w", padx=4)

        Label(frm, text="Max pages:").grid(row=4, column=0, sticky=E)
        Entry(frm, textvariable=self.max_pages, width=10).grid(row=4, column=1, sticky="w")

        Label(frm, text="Min cap len:").grid(row=4, column=2, sticky=E)
        Entry(frm, textvariable=self.min_cap_len, width=10).grid(row=4, column=3, sticky="w")

        Label(frm, text="Lang:").grid(row=5, column=0, sticky=E)
        Entry(frm, textvariable=self.lang, width=10).grid(row=5, column=1, sticky="w")

        Label(frm, text="Config:").grid(row=5, column=2, sticky=E)
        Entry(frm, textvariable=self.config, width=40).grid(row=5, column=3, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self.browse_config).grid(row=5, column=4)

        ttk.Button(frm, text="Load PDF", command=self.load_pdf).grid(row=6, column=0, pady=6)
        self.run_btn = ttk.Button(frm, text="Run", command=self.run_pipeline)
        self.run_btn.grid(row=6, column=1, pady=6, sticky="w")
        self.cancel_btn = ttk.Button(frm, text="Cancel", command=self.cancel_run, state="disabled")
        self.cancel_btn.grid(row=6, column=2, pady=6, sticky="w")

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(3, weight=1)

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=BOTH, expand=True)

        # PDF preview
        self.tab_pdf = ttk.Frame(self.nb)
        self.nb.add(self.tab_pdf, text="PDF Preview")
        self.tab_pdf.rowconfigure(0, weight=1)
        self.tab_pdf.columnconfigure(0, weight=0)
        self.tab_pdf.columnconfigure(1, weight=0)
        self.tab_pdf.columnconfigure(2, weight=1)
        self.preview_list = Listbox(self.tab_pdf)
        self.preview_list.grid(row=0, column=0, sticky="ns")
        list_scroll = ttk.Scrollbar(self.tab_pdf, orient="vertical", command=self.preview_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.preview_list.configure(yscrollcommand=list_scroll.set)
        self.preview_list.bind("<<ListboxSelect>>", self.on_page_select)
        self.canvas = ttk.Label(self.tab_pdf)
        self.canvas.grid(row=0, column=2, sticky="nsew")

        # Text preview
        self.tab_text = ttk.Frame(self.nb)
        self.nb.add(self.tab_text, text="Output Preview")
        self.tab_text.rowconfigure(0, weight=1)
        self.tab_text.columnconfigure(0, weight=1)
        self.text_area = Text(self.tab_text, wrap="word")
        self.text_area.grid(row=0, column=0, sticky="nsew")
        text_scroll = ttk.Scrollbar(self.tab_text, orient="vertical", command=self.text_area.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.text_area.configure(yscrollcommand=text_scroll.set)

    def _build_status(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill=X, padx=10, pady=(0, 8))
        self.progress = ttk.Progressbar(bar, mode="indeterminate")
        self.progress.pack(side=LEFT, fill=X, expand=True)
        self.status_label = ttk.Label(bar, text="Ready")
        self.status_label.pack(side=RIGHT)

    def browse_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.pdf_path.set(path)
            try:
                default_out = str(Path(path).resolve().parent / "out")
                self.out_dir.set(default_out)
            except Exception:
                pass

    def browse_out(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_dir.set(path)

    def browse_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml;*.yml"), ("All files", "*.*")])
        if path:
            self.config.set(path)

    def load_pdf(self) -> None:
        if not fitz:
            self.text_area.insert(END, "PyMuPDF not installed; PDF preview disabled.\n")
            return
        path = self.pdf_path.get()
        if not path:
            return
        try:
            self.doc = fitz.open(path)
        except Exception as exc:  # pragma: no cover
            self.text_area.insert(END, f"Failed to open PDF: {exc}\n")
            return
        self.preview_list.delete(0, END)
        for i in range(self.doc.page_count):
            self.preview_list.insert(END, f"Page {i+1}")
        if self.doc.page_count > 0:
            self.preview_list.select_set(0)
            self.on_page_select()

    def on_page_select(self, event=None) -> None:
        if not hasattr(self, "doc"):
            return
        sel = self.preview_list.curselection()
        if not sel:
            return
        page_idx = sel[0]
        page = self.doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25))
        # Convert to Tk photo image via PIL to support alpha
        from PIL import Image, ImageTk  # lazy

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.tk_img)

    def run_pipeline(self) -> None:
        # assemble args
        cmd = [sys.executable, str(Path(__file__).with_name("pdf2mm.py"))]
        if self.pdf_path.get():
            cmd += ["--pdf", self.pdf_path.get()]
        # Default outdir to PDF folder + /out if not set
        outdir_val = self.out_dir.get()
        if not outdir_val and self.pdf_path.get():
            try:
                outdir_val = str(Path(self.pdf_path.get()).resolve().parent / "out")
                self.out_dir.set(outdir_val)
            except Exception:
                outdir_val = str(Path.cwd() / "out")
                self.out_dir.set(outdir_val)
        if outdir_val:
            cmd += ["--outdir", outdir_val]
        cmd += ["--format", self.format.get() or "jsonl"]
        cmd += ["--ocr", self.ocr.get() or "auto"]
        cmd += ["--caption-provider", self.caption_provider.get() or "none"]
        cmd += ["--embed-provider", self.embed_provider.get() or "none"]
        cmd += ["--max-pages", self.max_pages.get() or "0"]
        cmd += ["--min-cap-len", self.min_cap_len.get() or "6"]
        if self.lang.get():
            cmd += ["--lang", self.lang.get()]
        if self.config.get():
            cmd += ["--config", self.config.get()]

        # Run in background to keep UI responsive
        def _run():
            try:
                self.set_running(True)
                self.status_label.config(text="Processing…")
                self.progress.start(10)
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = self.proc.communicate()

                def _update_preview():
                    # Load output preview (first few JSONL rows)
                    outdir = Path(self.out_dir.get())
                    jsonl_path = outdir / "data.jsonl"
                    self.text_area.delete("1.0", END)
                    if jsonl_path.exists():
                        lines = jsonl_path.read_text(encoding="utf-8").splitlines()[:50]
                        for ln in lines:
                            try:
                                obj = json.loads(ln)
                                unit = obj.get("unit")
                                page = obj.get("page")
                                text = obj.get("text") or obj.get("caption") or ""
                                text = (text[:200] + "…") if len(text) > 200 else text
                                self.text_area.insert(END, f"p{page} [{unit}] {text}\n")
                            except Exception:
                                self.text_area.insert(END, ln + "\n")
                    else:
                        self.text_area.insert(END, stderr or stdout or "No output produced")
                    self.status_label.config(text="Done")

                self.root.after(0, _update_preview)
            except Exception as exc:  # pragma: no cover
                def _err():
                    self.text_area.insert(END, f"Run failed: {exc}\n")
                    self.status_label.config(text="Error")
                self.root.after(0, _err)
            finally:
                self.root.after(0, self._stop_running)

        threading.Thread(target=_run, daemon=True).start()

    def set_running(self, running: bool) -> None:
        self.run_btn.config(state="disabled" if running else "normal")
        self.cancel_btn.config(state="normal" if running else "disabled")

    def _stop_running(self) -> None:
        self.progress.stop()
        self.set_running(False)
        self.proc = None

    def cancel_run(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass
            self.status_label.config(text="Canceled")


def main() -> int:
    root = Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

