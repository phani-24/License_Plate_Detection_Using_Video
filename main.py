import cv2
import pytesseract
import easyocr

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the input video
video = cv2.VideoCapture("5.mp4")

# Load the number plate cascade classifier
number_plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Create a list to store the text accuracies
texts_accuracies = []

# Loop through the frames of the input video
while True:
    # Read the next frame
    ret, frame = video.read()
    if not ret:
        # If there are no more frames, exit the loop
        break
        
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the number plates in the grayscale frame
    number_plates = number_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    
    # Loop through each detected number plate
    for (x, y, w, h) in number_plates:
        # Extract the number plate region from the frame
        plate = frame[y:y+h, x:x+w]
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Get text using EasyOCR
        result = reader.readtext(plate_gray)

        # Apply denoising, thresholding, and morphological closing to the number plate region
        plate_gray = cv2.fastNlMeansDenoising(plate_gray, None, 10, 7, 21)
        plate_bw = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        plate_bw = cv2.morphologyEx(plate_bw, cv2.MORPH_CLOSE, kernel)
        
        # Display the binary image
        cv2.imshow('Binary Image', plate_bw)
        
        # Get the text using Tesseract OCR
        tesseract_text = pytesseract.image_to_string(plate_gray, config='--psm 11')
        tesseract_text = ''.join([x for x in tesseract_text if x.isalnum()])

        # Get the text using Tesseract OCR with additional pre-processing
        tesseract_res=[x for x in tesseract_text if x != '\n' and x != ' ' and (x.isalpha() or x.isalnum())]
        number_plate_text = ''.join(tesseract_res)
        
        # If the Tesseract OCR result is longer than 3 characters, print it
        if len(number_plate_text)>3:
            print('From Tesseract - ', number_plate_text)
            
        # If EasyOCR detects any text
        if len(result)>0:
            # Extract the text from the first result and clean it up
            easy_res=[x for x in result[0][1] if x != '\n' and x != ' ' and (x.isalpha() or x.isalnum())]
            easy_res = ''.join(easy_res)
            
            # If there are more than one result, loop through them and extract the text
            if len(result)>1:
                for i in range(1,len(result)):
                    easy_res+=''.join([x for x in result[i][1] if x != '\n' and x != ' ' and (x.isalpha() or x.isalnum())])
        
            # Print the EasyOCR result
            print('From easyocr - ',easy_res," more abouts ",result)
            # Append the accuracy to the list
            if len(easy_res) > len(number_plate_text):
                diff = sum([ord(a) - ord(b) for a, b in zip(easy_res, number_plate_text)])
                number_plate_text =easy_res
                texts_accuracies.append((result[0][1],float(result[0][2]),diff))
                    
        # Displaying the frame surrounded by rectangular box
        if len(number_plate_text)>6:
             # Draw the rectangle box around the number plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
             # Display the recognized text
            cv2.putText(frame, number_plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Image", frame)
    
    # Exit the loop if the 'e' key is pressed
    if cv2.waitKey(1) == ord('e'):
        break
#sorting the obtained texts by accuracies and with comparing difference of tesseractocr and easyocr
text_accuracy_sorted = sorted(texts_accuracies, key=lambda x: x[1]/x[2] if x[2]!=0 else float('inf'))

#printing the most accurate one with accuracy
print('\n\nThe most accurate one that we can get is - ',text_accuracy_sorted[0][0],'\n And the accuracy is about',text_accuracy_sorted[0][1])
#Release the video capture and destroy all windows
video.release()
cv2.destroyAllWindows()
#Calculate and print the average text accuracy
if len(texts_accuracies) > 0:
    text_accuracy_sum = sum([x[1] for x in texts_accuracies])
    avg_accuracy = text_accuracy_sum/ len(texts_accuracies)
    print('And the Average text accuracy: {:.2f}%'.format(avg_accuracy*100))
else:
    print('No number plates detected.')