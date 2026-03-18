import cv2
import numpy as np
import pickle
import psycopg2

videoCap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

'Connecting to DataBase'
conn = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password='1234')

'Initializing cursor to execute commands in db'
cur = conn.cursor()



'This variable is need to append every x seconds face into the face_data array, so I have just enough data to recognize face'
skip = 0
face_data = []
dataset_path = './face_dataset/'

face_name = input('Enter the name of the person: \n')

while True:
    'here we are obtaining the camera feed, ret (boolean) tells if it is working, frame is next frame (actual image)'
    ret, frame = videoCap.read()

    if not ret:
        continue

    'in opencv we work with BGR instead of RGB and we need to translate it to gray scale'
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    'This line return the coordinates of the faces in the frame'
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

    if len(faces) == 0:
        continue

    k = 1

    '''Since our faces element looks like this [x,y,w,h] where w and h is width and height of the face,
    we are sorting array of faces by the area of the face in a decreasing order so we compute "bigger" faces first '''
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    skip += 1

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        'Our padding for face inside of a frame'
        face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset,(100,100))

        if skip % 5 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow(str(k), face_selection)
        k += 1

        'Here we put a rectangle on our face (x,y) stands for a top left corner and (x+w,y+h) stands for left bottom corner'
        'last arg is for thickness'
        "Also the y goes downwards that's why we adding h and not substracting, old things"
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('faces', frame)

    'waiting for pressing the "q" key to leave the loop, terminating the vide recording'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'converting to numpy array'
face_data = np.array(face_data)
'we reshape the data inside of an array from 3D to 1D shape, so KNN algorithm could work'
face_data = face_data.reshape(face_data.shape[0], -1) #-1 tells numpy to figure out the shape automatically
#print(face_data.shape)

'Serialiazing face_data into a byte-stream'
pickle_string = pickle.dumps(face_data)

'Part responsible for adding face_data to postgresql db'
cur.execute(f"""INSERT INTO faces (name, data) VALUES
({face_name}, {pickle_string})
""")


#np.save(dataset_path + face_name, face_data)
#print('Data saved at: {}'.format(dataset_path + face_name + '.npy'))

'Statements to end work with postgres'
conn.commit()
cur.close()
conn.close()

'End of work with cv2'
videoCap.release()
cv2.destroyAllWindows()
