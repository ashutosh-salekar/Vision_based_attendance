import cv2
import datetime
import pandas as pd
import numpy as np
import os
import face_recognition


### CSV file reading

attendence_file = r'/content/drive/MyDrive/05_Face_Recognition/Attendence.csv'
attendence = pd.read_csv(attendence_file)
rows, cols = attendence.shape
cols_names = attendence.columns


### Training

Train_img_folder = r'/content/drive/MyDrive/05_Face_Recognition/Train_Images'
Train_img_list = os.listdir(Train_img_folder)

Training_names = []
Training_encoded_imgs = []

for img in Train_img_list:
    Training_names.append(os.path.splitext(img)[0])

    read_img = cv2.imread(os.path.join(Train_img_folder,img))
    read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB) 
    Training_encoded_imgs.append(face_recognition.face_encodings(read_img)[0])


### Adding today's date

today_date = datetime.date.today()
if str(today_date) not in cols_names:
  attendence[str(today_date)] = np.zeros(rows)


### Add new employee names if required

Csv_names = attendence['Employee'].to_list()
names_to_add = list(set(Training_names).difference(Csv_names))

for new_name in names_to_add:
  attendence.loc[len(attendence.index)] = [new_name] + list(np.zeros(len(attendence.columns)-1))


### Testing

Test_file = '/content/drive/MyDrive/05_Face_Recognition/Test_Images/Virat_test_7.jpg'
Test_img = cv2.imread(Test_file)

## Function starts
Test_img_mod = cv2.resize(Test_img, (0,0), None, 0.25,0.25)
Test_img_mod = cv2.cvtColor(Test_img_mod, cv2.COLOR_BGR2RGB)


faces_in_frame = face_recognition.face_locations(Test_img_mod)
encoded_faces = face_recognition.face_encodings(Test_img_mod, faces_in_frame)


for encode_face, faceloc in zip(encoded_faces,faces_in_frame):

    matches = face_recognition.compare_faces(Training_encoded_imgs, encode_face)
    faceDist = face_recognition.face_distance(Training_encoded_imgs, encode_face)
    matchIndex = np.argmin(faceDist)
    
    if matches[matchIndex] and faceDist[matchIndex] <= 0.5:
        name = Training_names[matchIndex]
        y1,x2,y2,x1 = faceloc
        # since we scaled down by 4 times
        y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(Test_img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(Test_img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
        cv2.putText(Test_img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
        # Update CSV file
        row_to_update = np.where(attendence['Employee'] == name)[0][0]
        column_to_update = attendence.columns.get_loc(str(today_date))
        attendence.iloc[row_to_update, column_to_update] = 'Present'

attendence.to_csv(attendence_file,index=False)
print('CSV file update !!!')
