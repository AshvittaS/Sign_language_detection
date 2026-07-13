import cv2

print("Opening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened")
    raise SystemExit

print("Camera opened successfully")
print("Press Q inside the camera window to close")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()