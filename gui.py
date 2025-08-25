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

        self.pdf_path = StringVar()
        self.out_dir = StringVar(value=str(Path.cwd() / "out"))
        self.format = StringVar(value="jsonl")
        self.ocr = StringVar(value="auto")
        self.caption_provider = StringVar(value="none")
        self.embed_provider = StringVar(value="none")
        self.max_pages = StringVar(value="0")
        self.min_cap_len = StringVar(value="6")
        self.lang = StringVar(value="eng")
        self.config = StringVar(value="")

        self._build_controls()
        self._build_tabs()

    def _build_controls(self) -> None:
        frm = Frame(self.root)
        frm.pack(fill=X, padx=10, pady=6)

        Label(frm, text="PDF:").grid(row=0, column=0, sticky=E)
        Entry(frm, textvariable=self.pdf_path, width=60).grid(row=0, column=1, sticky="we", padx=4)
        Button(frm, text="Browse", command=self.browse_pdf).grid(row=0, column=2)

        Label(frm, text="Out dir:").grid(row=1, column=0, sticky=E)
        Entry(frm, textvariable=self.out_dir, width=60).grid(row=1, column=1, sticky="we", padx=4)
        Button(frm, text="Browse", command=self.browse_out).grid(row=1, column=2)

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
        Button(frm, text="Browse", command=self.browse_config).grid(row=5, column=4)

        Button(frm, text="Load PDF", command=self.load_pdf).grid(row=6, column=0, pady=6)
        Button(frm, text="Run", command=self.run_pipeline).grid(row=6, column=1, pady=6)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(3, weight=1)

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=BOTH, expand=True)

        # PDF preview
        self.tab_pdf = Frame(self.nb)
        self.nb.add(self.tab_pdf, text="PDF Preview")
        self.preview_list = Listbox(self.tab_pdf)
        self.preview_list.pack(side=LEFT, fill=Y)
        self.preview_list.bind("<<ListboxSelect>>", self.on_page_select)
        self.canvas = ttk.Label(self.tab_pdf)
        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)

        # Text preview
        self.tab_text = Frame(self.nb)
        self.nb.add(self.tab_text, text="Output Preview")
        self.text_area = Text(self.tab_text, wrap="word")
        self.text_area.pack(fill=BOTH, expand=True)

    def browse_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.pdf_path.set(path)

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
        if self.out_dir.get():
            cmd += ["--outdir", self.out_dir.get()]
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
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # Load output preview (first few JSONL rows)
                outdir = Path(self.out_dir.get())
                jsonl_path = outdir / "data.jsonl"
                if jsonl_path.exists():
                    lines = jsonl_path.read_text(encoding="utf-8").splitlines()[:50]
                    self.text_area.delete("1.0", END)
                    for ln in lines:
                        try:
                            obj = json.loads(ln)
                            # Render a human-friendly preview line
                            unit = obj.get("unit")
                            page = obj.get("page")
                            text = obj.get("text") or obj.get("caption") or ""
                            text = (text[:200] + "…") if len(text) > 200 else text
                            self.text_area.insert(END, f"p{page} [{unit}] {text}\n")
                        except Exception:
                            self.text_area.insert(END, ln + "\n")
                else:
                    self.text_area.insert(END, proc.stderr or proc.stdout)
            except Exception as exc:  # pragma: no cover
                self.text_area.insert(END, f"Run failed: {exc}\n")

        threading.Thread(target=_run, daemon=True).start()


def main() -> int:
    root = Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

