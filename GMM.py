# μ(t) = (1 - ρ)μ(t-1) + ρX(t)
# σ^2(t) = (1 - ρ)σ^2(t-1) + ρ(X(t) - μ(t))^T(X(t) - μ(t))
# ω(k, t) = (1 - α)ω(k, t-1) + αM(k, t)
import numpy as np
import pickle as pckl
import cv2
def initialize_means(pixel_values, num_components):
    random_indices = np.random.choice(len(pixel_values), size=num_components, replace=False)
    initial_means = pixel_values[random_indices]
    return initial_means

def initialize_weights(num_components):
    weights = np.random.dirichlet(np.ones(num_components), size=1)[0]
    return weights

def initialize_variances(pixel_values, num_components):
    overall_variance = np.var(pixel_values)

    initial_variances = np.full(num_components, overall_variance / num_components)
    return initial_variances

def GMM(pixel_values, learning_rate, num_components,initial_means=None, initial_weights=None, initial_variances=None):
    if initial_means is None:
        initial_means = initialize_means(pixel_values, num_components)
    if initial_weights is None:
        initial_weights = initialize_weights(num_components)
    if initial_variances is None:
        initial_variances = initialize_variances(pixel_values, num_components)

with open('histograms.pickle', 'rb') as f:
    histograms = pckl.load(f)

print(len(histograms[0]))

# temp = [value for inner_dict in histograms for value in inner_dict.values() if value > 0]
# print(temp)

# print(initialize_weights(5))
# print(initialize_variances(temp,5))
# print(initialize_means(temp,5))




# Open the video file
cap = cv2.VideoCapture("resources/vid2.mp4")

# Define the desired width and height for the output video
output_width = 640//2
output_height = 480//2

# Get the original video's width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (output_width, output_height))

# Read and process each frame of the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the desired resolution
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Write the resized frame to the output video file
    output_video.write(resized_frame)

    # Display the resized frame
    cv2.imshow("Resized Frame", resized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
output_video.release()

# # Close all OpenCV windows

cap = cv2.VideoCapture("output_video.mp4")
pixel_values={}
i=0
while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray_frame[45][30])
        #row-column
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()

cv2.destroyAllWindows()


num_components = 5
learning_rate = 0.01

cap = cv2.VideoCapture('resources/vid3.mp4')
ret, frame = cap.read() 
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
background = cv2.GaussianBlur(background, (5, 5), 0)  


gmm = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0) 

    fg_mask = gmm.apply(frame)

    threshold = 50 
    _, binary_mask = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('Foreground', binary_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
