# services/document_reader.py
# File parsing module with native PDF text extraction and offline Tesseract OCR fallback
import os
import sys
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

class DocumentReader:
    def __init__(self):
        self._configure_tesseract()

    def _configure_tesseract(self):
        """
        Attempts to locate Tesseract OCR binary. Checks environment variable
        TESSERACT_CMD first, then falls back to common system paths.
        """
        custom_path = os.getenv("TESSERACT_CMD")
        if custom_path and os.path.exists(custom_path):
            pytesseract.pytesseract.tesseract_cmd = custom_path
            return

        if sys.platform == "win32":
            # List common Windows Tesseract installation directories
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def read(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Routes the document to the correct parsing path based on MIME type.
        
        Args:
            file_bytes (bytes): Raw bytes of the document.
            mime_type (str): MIME type of the document.
            
        Returns:
            str: Consistently extracted raw text.
        """
        if mime_type == "text/plain":
            return self._read_plain_text(file_bytes)
        elif mime_type == "application/pdf":
            return self._read_pdf(file_bytes)
        elif mime_type.startswith("image/"):
            return self._read_image(file_bytes)
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

    def _read_plain_text(self, file_bytes: bytes) -> str:
        """Decodes raw text bytes using UTF-8."""
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            return file_bytes.decode("latin-1").strip()

    def _read_pdf(self, file_bytes: bytes) -> str:
        """
        Extracts text from PDF. Fast-path extracts native text. 
        If character count is below 50 (scanned document), falls back to OCR.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted_text = []
        
        # Native Text Extraction Pass
        for page in doc:
            extracted_text.append(page.get_text())

        full_text = "\n".join(extracted_text).strip()

        # Heuristic: If PDF contains almost zero characters, it is a scanned PDF image.
        # Fall back to page-by-page OCR rendering.
        if len(full_text) < 50:
            return self._ocr_pdf(doc)
            
        return full_text

    def _ocr_pdf(self, doc: fitz.Document) -> str:
        """Renders PDF pages as images in-memory and extracts text using Tesseract OCR."""
        ocr_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to high-quality image (zoom matrix for higher OCR accuracy)
            zoom = 2.0  # Increase resolution by 2x
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Read pixmap bytes into PIL Image in-memory
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Extract text from page image
            text = pytesseract.image_to_string(img)
            ocr_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
        return "\n".join(ocr_text).strip()

    def _read_image(self, file_bytes: bytes) -> str:
        """Reads image bytes directly into OCR engine."""
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img).strip()
