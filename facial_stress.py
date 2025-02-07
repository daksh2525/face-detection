import cv2
from deepface import DeepFace

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

    try:
        # Perform emotion analysis
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
        emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
        pass

    # Display the frame
    cv2.imshow("Facial Stress Detection", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
