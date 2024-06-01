from imutils import paths
import face_recognition
import pickle
import cv2
import os


imagePaths = list(paths.list_images('Dataset'))
knownEncodings = []
knownNames = []


print("=== Model learning started ===")

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    print(f"{name}: {imagePath}")

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)


data = {"encodings": knownEncodings, "names": knownNames}


f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()

print("=== Model is saved ===")