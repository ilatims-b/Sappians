from langdetect import detect,LangDetectException
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from pdf2image import convert_from_path
# import cv2
import pytesseract
from PIL import Image

# Extract image data
def extract_text_from_image(image_path):
    """Extract text from an image using pytesseract"""
    img = Image.open(image_path)
    custom_config = r'--psm 1'
    text = pytesseract.image_to_string(img,config=custom_config)
    if len(text)<5:
      return None
    return text


# Clean up the OCR text.
def clean_text(text):
  try:
    language = detect(text)
  except LangDetectException as e:
    print(f"Skipping file due to error: {str(e)}")

  text = text.lower()

  tokens = word_tokenize(text)

  stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
  tokens_cleaned = [word for word in tokens if word not in stop_words]

    # Keep only the first 30 words

  cleaned_text = ' '.join(tokens_cleaned[:30])
  return cleaned_text