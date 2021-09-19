import cv2
import os
dataset="dataset"
name="thyaanesh"
path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)
(width,height)=(1000,750)
alg="haarcascade_frontalface_default.xml"
haar_cascade=cv2.CascadeClassifier(alg)
vs=cv2.VideoCapture(0)
count=1
while count<50:
    print(count)
    _,img=vs.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceOnly=grayImg[y:y+h,x:x+w]
        resizeImg=cv2.resize(faceOnly,(width,height))
        cv2.imwrite("%s/%s.jpg"%(path,count),resizeImg)
        count+=1
    cv2.imshow("detection",img)
    key=cv2.waitKey(10)
    if key==ord("t"):
        break
print("captured successfully")
vs.release()
cv2.destroyAllWindows()
