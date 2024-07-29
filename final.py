import torch
import cv2
import numpy as np
# import android, time
# droid = android.Android()
# droid.startLocating()
# print('reading GPS ...')
# event=droid.eventWaitFor('location', 10000)
# while 1:
#     try :
#         provider = event.result['data']['gps']['provider']
#         if provider == 'gps':
#             lat = str(event['data']['gps']['latitude'])
#             lng = str(event['data']['gps']['longitude'])
#             latlng = 'lat: ' + lat + ' lng: ' + lng
#             print(latlng)
#             break
#        else: continue
#    except KeyError:
#        continue

# g = geocoder.ip('me')
# print(g.latlng)
model = torch.hub.load('yolov7', 'custom', path_or_model ='runs/train/YOLO_pothole25/weights/best.pt', source='local')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

