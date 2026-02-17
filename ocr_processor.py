# ocr_processor.py

import os
import io
import json
import pdfplumber
import numpy as np
from PIL import Image
import webcolors
from skimage.color import rgb2lab, deltaE_ciede2000


from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


class GoogleVisionOCRProcessor:

    # -------------------------------------------------------
    # Detect CID garbage
    # -------------------------------------------------------
    def looks_like_cid_encoded(self, text: str) -> bool:
        return "(cid:" in (text or "").lower()

    # -------------------------------------------------------
    # Build Vision Client
    # -------------------------------------------------------
    def _build_vision_client(self, user_type: str):
        user_type = (user_type or "").strip().lower()

        if user_type == "org":
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not service_account_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")

            credentials = service_account.Credentials.from_service_account_file(
                service_account_path
            )
            return vision.ImageAnnotatorClient(credentials=credentials)

        if user_type == "byok":
            return vision.ImageAnnotatorClient()

        raise ValueError("Invalid user_type. Use 'org' or 'byok'.")

    # -------------------------------------------------------
    # OCR Image Bytes
    # -------------------------------------------------------
    def ocr_image_bytes(
        self,
        image_bytes: bytes,
        client: "vision.ImageAnnotatorClient",
        language_hints=None
    ) -> str:

        if language_hints is None:
            language_hints = ["ja", "en"]

        image = vision.Image(content=image_bytes)
        image_context = vision.ImageContext(language_hints=language_hints)

        response = client.document_text_detection(
            image=image,
            image_context=image_context
        )

        if response.error.message:
            print(f"⚠️ OCR error: {response.error.message}")
            return ""

        annotation = response.full_text_annotation
        if annotation and annotation.text:
            return annotation.text.strip()

        return ""

    # -------------------------------------------------------
    # RGB → HEX
    # -------------------------------------------------------
    def rgb_to_hex(self, r, g, b):
        return "#{:02X}{:02X}{:02X}".format(r, g, b)

    # -------------------------------------------------------
    # Approximate Color Name
    # -------------------------------------------------------
        # -------------------------------------------------------
    # Accurate CSS3 Color Naming (Dynamic, No Hardcoding)
    # -------------------------------------------------------


    # -------------------------------------------------------
    # Professional Color Naming using LAB + DeltaE
    # -------------------------------------------------------
        # -------------------------------------------------------
    # Professional LAB Color Naming (Stable Version)
    # -------------------------------------------------------
    def get_color_name(self, r, g, b):

        try:
            return webcolors.rgb_to_name((r, g, b), spec="css3")
        except ValueError:
            pass

        # Convert input color to LAB
        target_rgb = np.array([[[r/255.0, g/255.0, b/255.0]]])
        target_lab = rgb2lab(target_rgb)

        min_distance = float("inf")
        closest_name = None

        for name in webcolors.names("css3"):
            hex_value = webcolors.name_to_hex(name, spec="css3")
            cr, cg, cb = webcolors.hex_to_rgb(hex_value)

            comparison_rgb = np.array([[[cr/255.0, cg/255.0, cb/255.0]]])
            comparison_lab = rgb2lab(comparison_rgb)

            distance = deltaE_ciede2000(target_lab, comparison_lab)[0][0]

            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name
    
        # -------------------------------------------------------
    # WCAG Contrast Ratio
    # -------------------------------------------------------
    def _relative_luminance(self, r, g, b):
        def channel(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        R = channel(r)
        G = channel(g)
        B = channel(b)

        return 0.2126 * R + 0.7152 * G + 0.0722 * B

    def contrast_ratio(self, rgb1, rgb2):
        L1 = self._relative_luminance(*rgb1)
        L2 = self._relative_luminance(*rgb2)

        lighter = max(L1, L2)
        darker = min(L1, L2)

        return round((lighter + 0.05) / (darker + 0.05), 2)

    # -------------------------------------------------------
    # Group Similar Colors (LAB + DeltaE)
    # -------------------------------------------------------
    def group_similar_colors(self, colors, threshold=15):

        groups = []
        used = set()

        for i, c1 in enumerate(colors):
            if i in used:
                continue

            group = [c1]
            used.add(i)

            rgb1 = np.array([[[c1["r"]/255, c1["g"]/255, c1["b"]/255]]])
            lab1 = rgb2lab(rgb1)

            for j, c2 in enumerate(colors):
                if j in used:
                    continue

                rgb2 = np.array([[[c2["r"]/255, c2["g"]/255, c2["b"]/255]]])
                lab2 = rgb2lab(rgb2)

                delta = deltaE_ciede2000(lab1, lab2)[0][0]

                if delta < threshold:
                    group.append(c2)
                    used.add(j)

            groups.append(group)

        return groups

    # -------------------------------------------------------
    # Build Professional Palette Report
    # -------------------------------------------------------
    def build_professional_color_report(self, dominant_colors):

        groups = self.group_similar_colors(dominant_colors)

        palette = []

        for group in groups:
            total_fraction = sum(c["pixel_fraction"] for c in group)
            representative = max(group, key=lambda x: x["pixel_fraction"])

            palette.append({
                "representative_hex": representative["hex"],
                "representative_name": representative["color_name"],
                "grouped_colors": [c["hex"] for c in group],
                "total_pixel_fraction": round(total_fraction, 3)
            })

        palette.sort(key=lambda x: x["total_pixel_fraction"], reverse=True)
        return palette




    # -------------------------------------------------------
    # Extract Dominant Colors (NO POSITION)
    # -------------------------------------------------------
    def extract_dominant_colors(
        self,
        image_bytes: bytes,
        client: "vision.ImageAnnotatorClient"
    ):

        image = vision.Image(content=image_bytes)
        response = client.image_properties(image=image)

        colors = []

        if response.image_properties_annotation:
            for color_info in response.image_properties_annotation.dominant_colors.colors:

                r = int(color_info.color.red)
                g = int(color_info.color.green)
                b = int(color_info.color.blue)

                colors.append({
                    "r": r,
                    "g": g,
                    "b": b,
                    "hex": self.rgb_to_hex(r, g, b),
                    "color_name": self.get_color_name(r, g, b),
                    "score": float(color_info.score),
                    "pixel_fraction": float(color_info.pixel_fraction)
                })

        return colors

    # -------------------------------------------------------
    # Full Image Analysis
    # -------------------------------------------------------
    def extract_full_image_analysis(
        self,
        image_bytes: bytes,
        client: "vision.ImageAnnotatorClient"
    ):

        image = vision.Image(content=image_bytes)
        result = {}

        # Colors
        result["dominant_colors"] = self.extract_dominant_colors(
            image_bytes=image_bytes,
            client=client
        )
        
                # 🔥 Add grouped professional palette (ADDITION ONLY)
        dominant_colors = result["dominant_colors"]

        result["color_palette_grouped"] = self.build_professional_color_report(
            dominant_colors
        )

        # 🔥 Add basic contrast analysis (ADDITION ONLY)
        if len(dominant_colors) >= 2:
            primary = dominant_colors[0]
            secondary = dominant_colors[1]

            result["contrast_analysis"] = {
                "primary_vs_secondary_ratio":
                    self.contrast_ratio(
                        (primary["r"], primary["g"], primary["b"]),
                        (secondary["r"], secondary["g"], secondary["b"])
                    )
            }


        # Labels
        label_response = client.label_detection(image=image)
        result["labels"] = [
            {"description": l.description, "score": float(l.score)}
            for l in getattr(label_response, "label_annotations", []) or []
        ]

        # Text Preview
        text_response = client.document_text_detection(image=image)
        fta = getattr(text_response, "full_text_annotation", None)

        if fta and getattr(fta, "text", None):
            result["text_preview"] = fta.text[:1500]

        return result

    # -------------------------------------------------------
    # Extract Full Page Images + Save Analysis JSON
    # -------------------------------------------------------
    def extract_images_only(
        self,
        pdf_path: str,
        output_folder: str,
        resolution: int = 500,
        user_type: str = "org"
    ):

        os.makedirs(output_folder, exist_ok=True)
        image_paths = []

        client = self._build_vision_client(user_type=user_type)

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):

                pdf_width = page.width
                pdf_height = page.height

                page_image = page.to_image(resolution=resolution)

                pil_image = page_image.original
                img_width, img_height = pil_image.size

                img_path = os.path.join(output_folder, f"page_{i+1}.png")
                page_image.save(img_path)
                image_paths.append(img_path)

                print(f"🖼 Saved image: {img_path}")
                print(f"📄 PDF Size: {pdf_width} x {pdf_height}")
                print(f"🖼 Image Size: {img_width} x {img_height}")

                img_bytes = io.BytesIO()
                page_image.save(img_bytes, format="PNG")

                analysis = self.extract_full_image_analysis(
                    image_bytes=img_bytes.getvalue(),
                    client=client
                )

                # 🔥 Add size metadata
                analysis["page_dimensions"] = {
                    "pdf_width_points": float(pdf_width),
                    "pdf_height_points": float(pdf_height),
                    "image_width_px": int(img_width),
                    "image_height_px": int(img_height)
                }

                json_path = img_path.replace(".png", "_analysis.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, indent=4, ensure_ascii=False)

                print(f"📊 Saved full analysis: {json_path}")

        return image_paths


    # -------------------------------------------------------
    # Extract Text From PDF
    # -------------------------------------------------------
    import os
    import io
    import pdfplumber

    def extract_text_from_pdf(
            self,
            pdf_path: str,
            user_type: str = "org",
            language_hints=None,
            output_dir: str = "extracted_images",
            keep_page_breaks: bool = True
        ) -> str:

        client = self._build_vision_client(user_type)

        # ✅ Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)

        # ✅ Get PDF base name (without extension)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # ✅ Create output txt path inside extracted_images2
        out_txt = os.path.join(output_dir, f"{pdf_name}_ocr.txt")

        all_pages_text = []

        with pdfplumber.open(pdf_path) as pdf:

            for i, page in enumerate(pdf.pages):

                print(f"\n📄 Processing Page {i + 1}/{len(pdf.pages)}")

                parts = []

                normal_text = (page.extract_text() or "").strip()

                if normal_text and self.looks_like_cid_encoded(normal_text):
                    normal_text = ""

                if normal_text:
                    parts.append(normal_text)

                else:
                    page_image = page.to_image(resolution=500)
                    img_bytes = io.BytesIO()
                    page_image.save(img_bytes, format="PNG")

                    ocr_result = self.ocr_image_bytes(
                        img_bytes.getvalue(),
                        client,
                        language_hints=language_hints
                    )

                    if ocr_result:
                        parts.append(ocr_result)

                combined = "\n\n".join(parts).strip()

                if combined:
                    all_pages_text.append(combined)

        final_text = (
            "\n\n".join(all_pages_text)
            if keep_page_breaks
            else "".join(all_pages_text)
        )

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(final_text)

        print("✅ Text saved to:", out_txt)

        return final_text


# should work for all formats