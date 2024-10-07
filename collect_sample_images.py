import os

import cv2

data_directory = 'D:\Machine Learning Projects\Sign-Language-Detection'

if not os.path.exists(data_directory):
    os.makedirs(data_directory)

number_of_classes = 3
sample_image_dataset_size = 100

cap = cv2.VideoCapture(0)

for number in range(number_of_classes):
    if not os.path.exists(os.path.join(data_directory, str(number))):
        os.makedirs(os.path.join(data_directory, str(number)))

    print('Collecting data for class {}'.format(number))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start!', (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    
    counter = 0

    while counter < sample_image_dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_directory, str(number), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()




