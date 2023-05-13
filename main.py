import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2 as cv
import pickle as pckl

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the video
        self.video_path = "resources/vid2.mp4"
        self.cap = cv.VideoCapture(self.video_path)

        # Create a label to display the video frames
        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        # Create a button labeled "Train"
        self.button = QPushButton("Train", self)
        self.button.setGeometry(10, 10, 100, 30)
        self.button.clicked.connect(self.train_callback)

        # Start the video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # Update frame every 33 milliseconds (30 fps)

        self.show()

    def train_callback(self):
        cap = self.cap
        print("Training...")
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        histograms={}
        frame_number=0
        # Read until video is completed
        # while(cap.isOpened()):
        # # Capture frame-by-frame
        #     if frame_number == 16:
        #         break

        #     ret, frame = cap.read()
        #     # Calculate histogram
        #     histograms[frame_number] = {pixel_value: 0 for pixel_value in range(256)}
            
        #     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #     height, width = gray_frame.shape
        #     for row in range(height):
        #         for col in range(width):
        #             # Get the pixel value at the current row and column
        #             pixel_value = gray_frame[row, col]
        #             print(frame_number,pixel_value)
        #             histograms[frame_number][pixel_value] += 1
        #             print(pixel_value)
        #     frame_number=frame_number+1
        # with open('histograms.pickle', 'wb') as f:
        #     pckl.dump(histograms, f)
        with open('histograms.pickle', 'rb') as f:
            histograms = pckl.load(f)
        print(histograms)
        #     if ret == True:
            
        #         # Display the resulting frame
        #         cv.imshow('Frame',frame)
            
        #         # Press Q on keyboard to  exit
        #         if cv.waitKey(25) & 0xFF == ord('q'):
        #             break
            
        #     # Break the loop
        #     else: 
        #         break
        
        # # When everything done, release the video capture object
        # cap.release()
        
        # Closes all the frames
        cv.destroyAllWindows()

    def update_frame(self):
        ret, frame = self.cap.read()

        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Convert the frame to QPixmap
            pixmap = QPixmap.fromImage(QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888))

            # Set the QPixmap as the label's image
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
