import cv2,numpy,os
haar='haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar)
datasets='dataset'
print("training")
(images,labels,names,id)=([],[],{},0)
for(subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
(images,labels)=[numpy.array(lis) for lis in [images,labels]]
print(images,labels)
(width,height)=(1000,750)
 
#model=cv2.face.LBPHFaceRecognizer_create()
model=cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
vs=cv2.VideoCapture(0)
cnt=0
while True:
    _,img=vs.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        if prediction[1]<800:
            cv2.putText(img,'%s-%.0f' %(names[prediction[0]],prediction[1]),(x-19,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(img,"unknown",(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if(cnt>100):
                print("unknown person")
                cv2.imwrite("unknown.jpg",img)
                cnt=0
    cv2.imshow("facerecognition",img)
    key=cv2.waitKey(10)
    if key==ord("t"):
        break
vs.release()
cv2.destroyAllWindows()
                                        
