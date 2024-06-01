import face_recognition, cv2
import pickle, os


cascPathface = os.path.dirname( cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

data = pickle.loads(open('face_enc', "rb").read())
video = cv2.VideoCapture("Video/video.mp4")

isSave = True
output_name = "Result.mp4"
fps_saving = 20

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen_resolution = (width, height)

if isSave:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    output = cv2.VideoWriter(output_name, fourcc, fps_saving, screen_resolution)

k = 3 # Коэффициент сжатия (влияет на скорость и качество)
compression_resolution = (width // k, height // k)

c = 10 # Раз в сколько кадров стоит выполнять обработку
counter = 0

print("=== Streaming start ===")


faces = None
names = None

while True:
    ret, frame = video.read()

    if not ret:
        print("ERROR: Camera not reading")
        break

    if counter <= 0:
        counter = c
        frame_compression = cv2.resize(frame, compression_resolution)

        gray = cv2.cvtColor(frame_compression, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

        encodings = face_recognition.face_encodings(frame_compression)
        names = []
        
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
    

            names.append(name)
    
    for ((x, y, w, h), name) in zip(faces, names):
        x, y, w, h = [i * k for i in [x, y, w, h]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    counter -= 1

    cv2.imshow("Frame", frame)
    if isSave:
        output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

if isSave:
    output.release()

video.release()
cv2.destroyAllWindows()