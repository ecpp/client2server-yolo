import requests
import cv2
from configparser import ConfigParser
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))
config_filepath = dir_path+"/config.ini"
exists = os.path.exists(config_filepath)
config = None
if exists:
    print("--------config.ini file found at ", config_filepath)
    config = ConfigParser()
    config.read(config_filepath)
else:
    print("---------config.ini file not found at ", config_filepath)
server_config = config["SERVER"]

server_ip=server_config["host"]
server_port=server_config["port"]
server_videostream_endpoint=server_config["videostream_endpoint"]
server_textstream_endpoint=server_config["textstream_endpoint"]

cap = cv2.VideoCapture(-1)
while True:
    success, img = cap.read()
    while not success:
        cap.release()
        cap=cv2.VideoCapture(0)
        success,img=cap.read()
    if success:    
        cv2.imshow("OUTPUT", img)
        _, imdata = cv2.imencode('.JPG', img)
        
        requests.put(server_ip+":"+server_port+server_videostream_endpoint, data=imdata.tobytes())
        
    if cv2.waitKey(40) == 27:  # 40ms = 25 frames per second (1000ms/40ms) 
        break

cv2.destroyAllWindows()
cap.release()
