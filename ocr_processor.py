import os
import io
import pdfplumber
from google.cloud import vision
from google.oauth2 import service_account

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

class GoogleVisionOCRProcessor:
    def looks_like_cid_encoded(self, text: str) -> bool:
        """Detects (cid:XX) patterns indicating CID-encoded garbage."""
        return "(cid:" in (text or "").lower()

    def ocr_image_bytes(
        self,
        image_bytes: bytes,
        client: "vision.ImageAnnotatorClient",
        language_hints=None
    ) -> str:
        """Calls Google Vision OCR on image bytes."""
        if language_hints is None:
            language_hints = ["ja", "en"]

        image = vision.Image(content=image_bytes)
        image_context = vision.ImageContext(language_hints=language_hints)
        response = client.document_text_detection(image=image, image_context=image_context)

        if getattr(response, "error", None) and response.error.message:
            print(f"⚠️ Google Vision OCR error: {response.error.message}")
            return ""

        annotation = getattr(response, "full_text_annotation", None)
        if annotation and getattr(annotation, "text", None):
            return annotation.text.strip()
        return ""

    def clamp_bbox_to_page(self, bbox, page_bbox):
        """Ensure the bbox is safely inside the page bbox."""
        x0, top, x1, bottom = bbox
        page_x0, page_top, page_x1, page_bottom = page_bbox

        x0 = max(page_x0, min(x0, page_x1))
        x1 = max(page_x0, min(x1, page_x1))
        top = max(page_top, min(top, page_bottom))
        bottom = max(page_top, min(bottom, page_bottom))

        if x0 >= x1 or top >= bottom:
            return None
        return (x0, top, x1, bottom)

    def _build_vision_client(self, user_type: str) -> "vision.ImageAnnotatorClient":
        """Create a Vision client based on user type."""
        user_type = (user_type or "").strip().lower()

        if user_type == "org":
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not service_account_path:
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS is not set. Put the service account JSON path in your .env or OS env."
                )
            credentials = service_account.Credentials.from_service_account_file(service_account_path)
            print("✅ Organization user - using service account from GOOGLE_APPLICATION_CREDENTIALS")
            return vision.ImageAnnotatorClient(credentials=credentials)

        if user_type == "byok":
            print("✅ BYOK user - using Application Default Credentials for Google Vision")
            return vision.ImageAnnotatorClient()

        raise ValueError("Invalid user_type. Use 'org' or 'byok'.")

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        user_type: str = "org",
        language_hints=None,
        out_txt: str = "ocr_extracted_text_new.txt",
        keep_page_breaks: bool = True
    ) -> str:
        """
        Extract text from PDF using text layer + selective OCR, with full-page OCR fallback.

        IMPORTANT:
        - Saves ONLY the extracted text (no '[FULL PAGE OCR]' / '[TEXT LAYER]' / page headers).
        """
        client = self._build_vision_client(user_type)

        all_pages_text = []
        ocr_pages_count = 0

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                print(f"\n📄 Processing Page {i + 1}/{len(pdf.pages)}")
                parts = []
                ocr_applied_this_page = False

                # 1) Try selectable text layer
                normal_text = (page.extract_text() or "").strip()
                if normal_text and self.looks_like_cid_encoded(normal_text):
                    print("⚠️ Detected CID-encoded garbage in text layer. Ignoring text layer.")
                    normal_text = ""

                has_text_layer = bool(normal_text)

                if has_text_layer:
                    print(f"✅ Found valid text layer on page {i + 1}")
                    parts.append(normal_text)  # ✅ no label

                    # 2) OCR embedded images only (logos, stamps, scanned snippets)
                    if page.images:
                        print(f"🔍 Found {len(page.images)} image(s) for OCR on page {i + 1}")
                        ocr_applied_this_page = True

                        page_bbox = (0.0, 0.0, float(page.width), float(page.height))
                        for img_idx, img in enumerate(page.images):
                            try:
                                raw_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                                safe_bbox = self.clamp_bbox_to_page(raw_bbox, page_bbox)
                                if not safe_bbox:
                                    print(f"⚠️ Skipping invalid bbox: {raw_bbox}")
                                    continue

                                cropped_img = page.crop(safe_bbox).to_image(resolution=500)
                                img_bytes = io.BytesIO()
                                cropped_img.save(img_bytes, format="PNG")

                                ocr_result = self.ocr_image_bytes(
                                    img_bytes.getvalue(), client, language_hints=language_hints
                                ).strip()

                                if ocr_result:
                                    parts.append(ocr_result)  # ✅ no label
                            except Exception as e:
                                print(f"⚠️ Error OCR-ing image #{img_idx + 1} on page {i + 1}: {e}")

                else:
                    # 3) If no text layer → full-page OCR
                    print(f"⚠️ No selectable text on page {i + 1}. Performing full-page OCR.")
                    ocr_applied_this_page = True

                    try:
                        page_image = page.to_image(resolution=500)
                        img_bytes = io.BytesIO()
                        page_image.save(img_bytes, format="PNG")

                        full_page_ocr_result = self.ocr_image_bytes(
                            img_bytes.getvalue(), client, language_hints=language_hints
                        ).strip()

                        if full_page_ocr_result:
                            parts.append(full_page_ocr_result)  # ✅ no label
                    except Exception as e:
                        print(f"⚠️ Error during full-page OCR on page {i + 1}: {e}")

                if ocr_applied_this_page:
                    ocr_pages_count += 1

                combined = "\n\n".join([p for p in parts if p]).strip()
                if combined:
                    all_pages_text.append(combined)
                else:
                    print(f"⚠️ No text found at all on page {i + 1}")

        print(f"\n📊 OCR was applied on {ocr_pages_count} out of {len(pdf.pages)} pages.")

        # ✅ Save ONLY text (no PAGE headers, no OCR labels)
        final_text = ("".join(all_pages_text)) if keep_page_breaks else ("\n\n".join(all_pages_text))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(final_text)

        print("✅ Saved to:", out_txt)
        return final_text
