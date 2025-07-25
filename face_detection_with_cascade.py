import cv2

face_cascade = cv2.CascadeClassifier("FaceDetection\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("FaceDetection\\haarcascade_eye.xml")

while True : 
    img = cv2.imread('FaceDetection\\photo.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces : 
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_img = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)

        for (ex, ey, ew, eh) in eyes :
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (8, 8, 8), 2)
    
    cv2.imshow("img", img)

    if cv2.waitKey(30) & 0xFF == 27 : 
        break

cv2.destroyAllWindows()