import cv2
import numpy as np
import face_recognition


imgaman = face_recognition.load_image_file("images/Aman Negi.jpg")
imgaman = cv2.cvtColor(imgaman,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file("images/AN.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgaman)[0]
encodeAman = face_recognition.face_encodings(imgaman)[0]
cv2.rectangle(imgaman,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
print(faceLocTest)
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255), 2)

result = face_recognition.compare_faces([encodeAman], encodeTest)
facedis = face_recognition.face_distance([encodeAman], encodeTest)
print(result)
print(facedis)

imS = cv2.resize(imgaman, (360, 540))
imD = cv2.resize(imgtest, (360, 540))

cv2.imshow("Aman Negi",imS)
cv2.imshow("Aman Test",imD)
cv2.waitKey(0)


