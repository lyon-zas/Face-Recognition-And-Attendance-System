import cv2
import numpy as np
import dlib 
import face_recognition

img = face_recognition.load_image_file("")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
imgTest = face_recognition.load_image_file("")
imgRGBTest = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

facelocation = face_recognition.face_locations(img)[0]
encodeImg =  face_recognition.face_encoding(img)[0]
cv2.rectangle(img,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2],(255,0,255),2))

facelocationTest = face_recognition.face_locations(imgRGBTest)[0]
encodeImgTest =  face_recognition.face_encoding(imgRGBTest)[0]
cv2.rectangle(img,(facelocationTest[3],facelocationTest[0]),(facelocationTest[1],facelocationTest[2],(255,0,255),2))

results = face_recognition.compare_faces([encodeImg],encodeImgTest)
faceDis = face_recognition.face_distances([encodeImg],encodeImgTest)
print(results,faceDis)

cv2.putText(imgTest,f"{results} {round(faceDis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
while True:
    success, img = img.read()

    cv2.imshow("Image", img)
    cv2.imshow("ImageTest", imgTest)

    cv2.waitKey(0)