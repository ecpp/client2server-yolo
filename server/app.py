import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from camera import Camera
from io import BytesIO
import torch
from flask import Flask, render_template, request, redirect, Response
import json

app = Flask(__name__)
frame = None


model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
        )#.autoshape()  # force_reload = recache latest code
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 



def gen():
    global frame
    while True:
        img = Image.open(io.BytesIO(frame))
        results = model(img, size=640)
        #results.print()  # print results to screen
        img = np.squeeze(results.render()) #RGB
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
        num_of_object = len(results.xyxy[0])
        num_of_people = 0
        for i in range(num_of_object):
            if results.xyxy[0][i][5] == 0:
                num_of_people += 1
        #display number of objects detected on screen
        cv2.putText(img_BGR, "Number of objects detected: " + str(num_of_object), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #display number of people detected on screen
        cv2.putText(img_BGR, "Number of people detected: " + str(num_of_people), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        frame2 = cv2.imencode('.jpg', img_BGR)[1].tobytes()       
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

def gen2():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + frame + b'\r\n')

@app.route('/upload', methods=['PUT'])
def upload():
    global frame
    frame = request.data
    return "OK"

@app.route('/watch')
def video():
    if frame:
        # if you use `boundary=other_name` then you have to yield `b--other_name\r\n`
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ""


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/text_stream', methods=['POST'])
def text_stream():
    data = request.data
    num_of_people = 0
    aList = json.loads(data)
    for i in range(len(aList)):
        if aList[i] == "person":
            num_of_people += 1
    print("Number of people detected: " + str(num_of_people))
    return str(num_of_people)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="91.191.173.36", port=args.port)
