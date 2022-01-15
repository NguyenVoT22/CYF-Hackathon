import cv2
import numpy as np
import os 
import smtplib
from email.message import EmailMessage
# from sense_hat import SenseHat
from sense_emu import SenseHat

# ⊱ ────── {Anvil imports that never worked} ────── ⊰
#╔══════ ≪≫°✺°≪ ≫ ══════╗

# import anvil.server
# from anvil.tables import app_tables
# from time import sleep
# from datetime import datetime

#╚══════ ≪≫°✺°≪ ≫ ══════╝

r = (255, 0, 0) #red
g = (0, 255, 0) #green
b = (0, 0, 255) #blue
e = (0, 0, 0) #empty
w = (255, 255, 255) #white
c = (0, 255, 255) #cyan
y = (255, 255, 0) #yellow
o = (255, 128, 0) #orange
n = (255, 128, 128) #pink
p = (128, 0, 128) #purple
d = (255, 0, 128) #darkPink
l = (128, 255, 128) #lightGreen
s = (255,105,97) # strawberry
m = (111, 153, 35) # matcha


sense = SenseHat()

matcha_heart = [
e, e, e, e, e, e, e, e,
e, m, m, e, m, m, e, e,
m, m, m, m, m, m, m, e,
m, m, m, m, m, m, m, e,
m, m, m, m, m, m, m, e,
e, m, m, m, m, m, e, e,
e, e, m, m, m, e, e, e,
e, e, e, m, e, e, e, e
]

strawberry_heart = [
e, e, e, e, e, e, e, e,
e, s, s, e, s, s, e, e,
s, s, s, s, s, s, s, e,
s, s, s, s, s, s, s, e,
s, s, s, s, s, s, s, e,
e, s, s, s, s, s, e, e,
e, e, s, s, s, e, e, e,
e, e, e, s, e, e, e, e
]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#initiate id counter
id = 0

# Update these names. Make sure to add as many as IDs trained; assumes first ID is 0.
names = ['You', 'Older Sibling', 'Tiny Sibling'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            # Email saying yesh this is your fam member
            #build the email with the EmailMessage() object.
            msg = EmailMessage()
                
            msg['Subject'] = "Sibling Passes By"
            msg['From'] = "otheremail.locked@gmail.com"
            msg['To'] = "otheremail.locked@gmail.com"

            #set the body with .set_content()
            msg.set_content("Chill ok?\nThis is just either you, older sibling, or younger sibling chill.")

            #use smptlib to send using gmail server, on port 465. 
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

            #replace with your own credentials
            server.login("otheremail.locked@gmail.com", "Codenext123")
                
            #send email   
            server.send_message(msg)
                    
            print("sending sibling confirmation")
                    
            server.quit()   

            with open('siblings.txt', 'a+') as f:    
                f.write("Yo it's your siblings.")
                f.write("\n")

            sense.set_pixels(matcha_heart)

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            # Email saying yesh this is me/any other intruder
            #build the email with the EmailMessage() object.
            msg = EmailMessage()
                
            msg['Subject'] = "Intruder?!"
            msg['From'] = "otheremail.locked@gmail.com"
            msg['To'] = "otheremail.locked@gmail.com"

            #set the body with .set_content()
            msg.set_content("Is that your parents..?\nBro someone is at your door/computer stay safe..")

            #use smptlib to send using gmail server, on port 465. 
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

            #replace with your own credentials
            server.login("otheremail.locked@gmail.com", "Codenext123")
                
            #send email   
            server.send_message(msg)
                    
            print("Sending email...")
                    
            server.quit()   


            with open('parents.txt', 'a+') as f:    
                f.write("Yo your parents are here. Or is it someone else?")
                f.write("\n")
            
            sense.set_pixels(strawberry_heart)
               
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

#♡━━━━━━ ◦ Anvil Code that never worked ◦ ━━━━━━♡
#┌─── ･ ｡ﾟ★: *.☪ .* :☆ﾟ. ───┐

# anvil.server.connect("GSAONDRKDQ2KQEJNCVWWWLI2-NIWB4MQ6WZ4OPU3X-CLIENT")


# @anvil.server.callable
# def get_env_data():
    
#     humidity = sense.get_humidity()
#     temperature = sense.get_temperature()
#     pressure = sense.get_pressure()
    
#     data = {
#         "h":humidity,
#         "t": temperature,
#         "p": pressure
        
#         }
#     return data

# @anvil.server.callable
# def save_data():
    
#     data = get_env_data()
#     t = data["t"]
#     h = data["h"]
#     p = data["p"]

#     dt = datetime.now()
    
#     new_row = app_tables.env_data.add_row(datetime=dt,
#                                           temperature=t,
#                                           humidity=h,
#                                           pressure=p)
    
#     print(new_row)

# while True:
#     save_data()
#     sleep(10)

#└─── ･ ｡ﾟ★: *.☪ .* :☆ﾟ. ───┘