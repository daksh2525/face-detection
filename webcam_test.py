import cv2
from deepface import DeepFace

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]  # Crop face region
        
        try:
            # Perform emotion analysis on the detected face
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display detected emotion
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")

    # Show output
    cv2.imshow("Facial Stress Detection", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
