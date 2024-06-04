import cv2
import torch
import os

print(torch.cuda.is_available())

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Using device:", device)
print("Model device:", next(model.parameters()).device)
model.to(device)
print(torch.cuda.get_device_name(0)) 


def process_video(video_path):

    print("Ruta absoluta del video:", os.path.abspath(video_path))

    cap = cv2.VideoCapture(video_path) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video terminado.")
            break
        
        frame = frame[:, :, ::-1] 
        
        results = model(frame)
        
        frame = results.render()[0]
        frame = frame[:, :, ::-1]  

        cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(100) == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows() 

process_video('Autos.MOV')