from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR()

def extract_license_plate(frame, x1, y1, x2, y2):
    plate_roi = frame[y1:y2, x1:x2]
    result = ocr.ocr(plate_roi, cls=True)
    
    plate_text = ""
    for line in result:
        for word in line:
            plate_text += word[1][0] + " "
    
    return plate_text.strip()

# Example Usage
frame = cv2.imread("car_with_plate.jpg")
x1, y1, x2, y2 = 100, 200, 300, 250  # Example plate position
plate_number = extract_license_plate(frame, x1, y1, x2, y2)
print("Detected License Plate:", plate_number)
