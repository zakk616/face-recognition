import cv2
import numpy as np
import face_recognition

img1 = face_recognition.load_image_file(
    r"D:\\GIL\\FaceRecognition\\imgs\\zakk1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(img1)[0]
encodeElon = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (faceLoc[3], faceLoc[0],
                     faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

img2 = face_recognition.load_image_file(
    r"D:\\GIL\\FaceRecognition\\imgs\\zakk2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
testLoc = face_recognition.face_locations(img2)[0]
encodeTest = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2, (testLoc[3], testLoc[0],
                     testLoc[1], testLoc[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeElon], encodeTest)

print(result)
cv2.imshow("img2", img2)
cv2.imshow("img1", img1)
cv2.waitKey(0)
