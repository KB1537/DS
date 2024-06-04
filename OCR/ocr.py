import cv2
import pytesseract
from PIL import Image
import sys
import os
import io
from pdf2image import convert_from_path
 
PAPER_FILE = "./WID/menopause_report.pdf"
pil_image_lst = convert_from_path(PAPER_FILE)
print(pil_image_lst)
pil_image = pil_image_lst[120]
 
print(pil_image)
 
boxes = pytesseract.image_to_string(pil_image)
 
print(boxes)