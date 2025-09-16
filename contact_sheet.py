# contact_sheet.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact sheet utilities (PNG/PDF, even sampling)
pip install pillow
"""

from __future__ import annotations
import math, os
from io import BytesIO
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

def sample_evenly(paths: List[str], limit: int) -> List[str]:
    if not paths:
        return []
    if limit <= 0 or limit >= len(paths):
        return paths
    step = max(1, len(paths) // limit)
    return paths[::step][:limit]

def make_contact_sheet(
    image_paths: List[str],
    cols: int = 5,
    thumb_width: int = 300,
    pad: int = 8,
    bg: Tuple[int, int, int] = (255, 255, 255),
    annotate: bool = False,
    font_path: Optional[str] = None,
    font_size: int = 16,
    draw_border: bool = False,
) -> Image.Image:
    if not image_paths:
        raise ValueError("image_paths is empty")

    thumbs: List[Tuple[str, Image.Image]] = []
    for p in image_paths:
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        new_h = max(1, int(h * (thumb_width / float(w))))
        im = im.resize((thumb_width, new_h), Image.LANCZOS)
        thumbs.append((os.path.basename(p), im))
    if not thumbs:
        raise ValueError("no readable images")

    cols = max(1, int(cols))
    rows = math.ceil(len(thumbs) / cols)

    row_heights: List[int] = []
    for r in range(rows):
        row_imgs = [im for _, im in thumbs[r*cols:(r+1)*cols]]
        row_heights.append(max(img.size[1] for img in row_imgs))

    sheet_w = cols * thumb_width + (cols + 1) * pad
    sheet_h = sum(row_heights) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)

    draw = ImageDraw.Draw(sheet)
    font = None
    if annotate:
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    y = pad
    idx = 0
    border_color = (200, 200, 200)
    label_bg = (255, 255, 255)
    label_fg = (0, 0, 0)

    for r in range(rows):
        x = pad
        for c in range(cols):
            if idx >= len(thumbs): break
            name, im = thumbs[idx]
            sheet.paste(im, (x, y))
            if draw_border:
                draw.rectangle([x, y, x+im.size[0]-1, y+im.size[1]-1], outline=border_color, width=1)
            if annotate:
                lh = font_size + 6
                ly1 = y + im.size[1] - lh; ly2 = y + im.size[1]
                draw.rectangle([x, ly1, x+im.size[0], ly2], fill=label_bg)
                draw.text((x+4, ly1+3), name, fill=label_fg, font=font)
            x += thumb_width + pad
            idx += 1
        y += row_heights[r] + pad

    return sheet

def build_contact_sheet_bytes(
    image_paths: List[str],
    cols: int = 5,
    thumb_width: int = 300,
    pad: int = 8,
    bg: Tuple[int, int, int] = (255, 255, 255),
    annotate: bool = False,
    font_path: Optional[str] = None,
    font_size: int = 16,
    draw_border: bool = False,
    pdf_resolution: float = 150.0,
) -> Tuple[bytes, bytes]:
    sheet = make_contact_sheet(
        image_paths, cols=cols, thumb_width=thumb_width, pad=pad, bg=bg,
        annotate=annotate, font_path=font_path, font_size=font_size, draw_border=draw_border
    )
    png_buf = BytesIO(); sheet.save(png_buf, format="PNG"); png_bytes = png_buf.getvalue()
    pdf_buf = BytesIO(); sheet.convert("RGB").save(pdf_buf, format="PDF", resolution=pdf_resolution)
    return png_bytes, pdf_buf.getvalue()

def save_contact_sheet_png(image_paths: List[str], out_path: str, **kwargs) -> str:
    img = make_contact_sheet(image_paths, **kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path

def save_contact_sheet_pdf(image_paths: List[str], out_path: str, pdf_resolution: float = 150.0, **kwargs) -> str:
    img = make_contact_sheet(image_paths, **kwargs).convert("RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PDF", resolution=pdf_resolution)
    return out_path

def list_images_in_dir(directory: str, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]
    files.sort()
    return files

def make_contact_sheet_from_dir(
    directory: str,
    out_png: Optional[str] = None,
    out_pdf: Optional[str] = None,
    limit: Optional[int] = None,
    stride: Optional[int] = None,
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    paths = list_images_in_dir(directory)
    if stride and stride > 1:
        paths = paths[::stride]
    if limit:
        paths = sample_evenly(paths, limit)
    png_path = save_contact_sheet_png(paths, out_png, **kwargs) if out_png else None
    pdf_path = save_contact_sheet_pdf(paths, out_pdf, **kwargs) if out_pdf else None
    return png_path, pdf_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        in_dir, out_png, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3]
        make_contact_sheet_from_dir(
            in_dir, out_png=out_png, out_pdf=out_pdf,
            limit=60, cols=6, thumb_width=280, annotate=False, draw_border=True
        )
        print(f"[OK] Saved: {out_png}, {out_pdf}")
    else:
        print("Usage: python contact_sheet.py <frames_dir> <out_png> <out_pdf>")
