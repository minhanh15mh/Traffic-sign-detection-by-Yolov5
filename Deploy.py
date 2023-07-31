import cv2
import torch
import pytesseract
 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_result(img_direct, model):
  result = model(img_direct)
  label = result.xyxy[0][:, -1]
  coordinate = result.xyxy[0][:, :-1]
  return label, coordinate

def crop_text_region(coordinate, img_direct):
  
  image = cv2.imread(img_direct)

  xmin, ymin, xmax, ymax, confident = coordinate
  x = int(xmin) # xmin
  y = int(ymin) # ymin
  w = int(xmax - xmin) # xmax - xmin
  h = int(ymax - ymin) # ymax - ymin
  
  start_point = (int(xmin), int(ymin))
  end_point = (int(xmax), int(ymax))  
  text_region = image[y:y+h, x:x+w]
  return text_region, start_point, end_point

def OCR(text_region):
  text = pytesseract.image_to_string(text_region, lang = 'vie')
  return print(text)

def plot_bboxes(start_point, end_point, text_region, img_direct):
  image = cv2.imread(img_direct)
  image = cv2.rectangle(image, start_point, end_point,color=(0,255,0), thickness=2)
  text =  OCR(text_region)
  image = cv2.putText(image, text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
  return cv2.imshow(image)

def main():  
  img_direct = r".\sign_test.jpg"
  model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
  
  label, coordinate = detect_result(img_direct, model_yolo)
  class_text = [1,2,4,5,7,8,10,11,13,14,17,19,20,22,23]
  
  text_region = []
  start_point = []
  end_point = []
  for i in range(len(label)):
    if label[i] in class_text:
       text_region_i, start_point_i, end_point_i = crop_text_region(coordinate[i], img_direct)
       text_region.append(text_region_i), start_point.append(start_point_i), end_point.append(end_point_i)

  for k in range(len(text_region)):
     plot_bboxes(start_point[k], end_point[k], text_region[k], img_direct)

if __name__ == "__main__":
    main()

