import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import psycopg2

'Initializng knn module from sklearn'
knn = KNeighborsClassifier(n_neighbors=5)

'Connecting to DataBase'
conn = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password='1234')

'Initializing cursor to execute commands in db'
cur = conn.cursor()


videoCap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cur.execute("SELECT id, name, data FROM faces")
rows = cur.fetchall()
conn.commit()

face_data = []
labels = []
class_id = 0
names = {} #mapping between id and name

'dataset preparation'

for row in rows: #we are looping through dataset directory
    face_id, name, data = row
    names[class_id] = name
    'Deserializng object'
    data_item = pickle.loads(data)

    face_data.append(data_item)
    'This line creates an array of labels for a given face, where every element is the class_id of that person'
    target = class_id * np.ones((data_item.shape[0,]))
    class_id += 1
    labels.append(target)

'''After np.concatenate at the end, all labels are joined into one flat array
which is used to train a classifier to associate face embeddings with a person's ID.
'''
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1,1)

'Training model'
knn.fit(face_dataset, face_labels.ravel())

while True:
    'here we are obtaining the camera feed, ret (boolean) tells if it is working, frame is next frame (actual image)'
    ret, frame = videoCap.read()

    if not ret:
        continue

    'in opencv we work with BGR instead of RGB and we need to translate it to gray scale'
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    'This line return the coordinates of the faces in the frame'
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        offset = 5
        'Our padding for face inside of a frame'
        face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_offset, (100, 100))
        'Model predicting logic'
        out = knn.predict(face_section.flatten().reshape(1, -1))
        predicted_name = names[int(out[0])]

        cv2.putText(frame, predicted_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'Statements to end work with postgres'
cur.close()
conn.close()

'End of work with cv2'
videoCap.release()
cv2.destroyAllWindows()




