import cv2

def select_object(frame):
    print("Select the object to track and press ENTER or SPACE.")
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    return bbox

def main():
    video_path = r"C:\Users\kbtod\Downloads\IMG_2272.MOV"  # Use 0 for webcam, or provide a file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Select initial object to track
    bbox = select_object(frame)
    if bbox == (0, 0, 0, 0):
        print("No object selected. Exiting...")
        return

    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker.init(frame, bbox)

    print("Tracking started. Press 'q' to quit, 'r' to reinitialize.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("MOSSE Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Reinitializing tracker...")
            bbox = select_object(frame)
            if bbox != (0, 0, 0, 0):
                tracker = cv2.legacy.TrackerMOSSE_create()
                tracker.init(frame, bbox)
                print("Tracker reinitialized.")
            else:
                print("Invalid selection. Continuing...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
