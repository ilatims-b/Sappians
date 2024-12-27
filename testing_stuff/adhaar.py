from pdf2image import convert_from_path

# Convert PDF to images
pages = convert_from_path("adhaar_sample.pdf", dpi=300)  # Higher DPI improves OCR accuracy
for i, page in enumerate(pages):
    page.save(f"page_{i}.jpg", "JPEG")

import pytesseract
from PIL import Image

# Perform OCR
text = pytesseract.image_to_string(Image.open("page_0.jpg"), lang="eng")
print(text)

