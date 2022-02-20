from django.shortcuts import render
import cv2
import numpy as np

# Create your views here.

def predict_digit(img):
   # resize image to 28x28 pixels
   img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
   img = img.reshape(1, 28, 28, 1)
   # normalizing the image to support our model input
   img = img / 255.0
   # predicting the class
   res = DigitrecappConfig.digitmodel.predict([img])[0]
   return np.argmax(res), max(res)

def classify_handwriting(image):
   img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
   print(img.shape)
   # converting to grayscale
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # apply thresholding
   ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV,      cv2.THRESH_OTSU)
   # find the contours
   contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
   for cnt in contours:
      # get bounding box and exact region of interest
      x, y, w, h = cv2.boundingRect(cnt)
      # create rectangle
      cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
      top = int(0.05 * th.shape[0])
      bottom = top
      left = int(0.05 * th.shape[1])
      right = left
      th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
      # Extract the image's region of interest
      roi = th[y - top : y + h + bottom, x - left : x + w + right]
      digit, acc = predict_digit(roi)
      return digit, acc

@api_view(["POST"])  
def getimagefromrequest(request):
   image = request.FILES.get("file")
   image_bytes = image.read()
   digit, acc = classify_handwriting(image_bytes)
   print(str(digit))
   return JsonResponse({"digit": str(digit), "accuracy": str(acc)})