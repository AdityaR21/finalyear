# importing needed libraries

from flask import Flask, render_template, request, flash, redirect, url_for, Response, send_file, stream_with_context, \
    jsonify
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from main import g1
import pyotp
import cv2
import numpy as np
import os

import cv2

# from numpy import asarray
from keras.models import load_model




# import base64
# from io import BytesIO
# import json
# import random
from keras.applications.vgg16 import preprocess_input
import cv2
import dlib
from math import hypot
import numpy as np
from numpy import expand_dims



from flask_cors import CORS
import qrcode
from io import BytesIO

# import pyqrcode

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import time
import cv2
import numpy as np
import dlib
from math import hypot

start_time=time.time()

count1 = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=20) #default image size = 160 #postprocess reduces accuracy.
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=20)# keep_all=True #keep all is to when there are multiple faces in frames in webcam
# margin adds a margin to detected faces

resnet = InceptionResnetV1(pretrained='vggface2').eval()
font = cv2.FONT_HERSHEY_PLAIN
#print(resnet)
dataset = datasets.ImageFolder('train') # photos folder path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names
# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data1.pt')
embedding_list = load_data[0]
name_list = load_data[1]


# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye blink detection functions
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Func to get midpoint of eye ratio
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Func to get bliking ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 255), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 255), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght

    return ratio


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# configuring flask application
app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "APP_SECRET_KEY"

app.config['SECRET_KEY'] = '\xff\x98Tq\x80\xf3\xb6\xac=\x10\xbfG\x9a\x98\x1e\xab098fi\xed\x1d@'

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:finalyr@localhost:5432/final_year_project_database'

db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = 'new_new2_users'
    name = db.Column(db.String(length=30))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(length=20))
    address = db.Column(db.String(length=255))
    secret = db.Column(db.String(length=32))
    phone_number = db.Column(db.String(length=14), primary_key=True)


db.Model.metadata.create_all(db.engine)

Bootstrap(app)


# homepage route
@app.route("/")
def index():
    return render_template("home.html")


# login page route
@app.route("/login/")
def login():
    return render_template("login.html")


@app.route("/trial_login/")
def trail_login():
    return render_template("trial_login.html")


# login form route
@app.route("/login/", methods=["POST"])
def login_form():
    # demo creds
    # creds = {"mobile_number": "+91 9597461340"}
    # getting form data
    mobile_number = request.form.get("mobile_number")

    # authenticating submitted creds with demo creds
    # if username == creds["username"] and password == creds["password"]:
    verify_user = db.session.query(User).filter(User.phone_number == mobile_number).first()
    if verify_user == None:
        flash("Invalid number", "danger")
        return redirect(url_for("login"))
    else:
        flash("You're current account holder", "success")
        return redirect(url_for("login_2fa", mobile_number=mobile_number))

    '''if mobile_number == creds["mobile_number"]:
        # inform users if creds are valid


        flash("You're current account holder", "success")
        return redirect(url_for("login_2fa"))
    else:
        # inform users if creds are invalid
        flash("Create an account!", "danger")
        return redirect(url_for("login"))
    '''


@app.route("/trial_login/", methods=["POST"])
def trial_login_form():
    # demo creds
    # creds = {"mobile_number": "+91 9597461340"}
    # getting form data
    mobile_number = request.form.get("mobile_number")

    # authenticating submitted creds with demo creds
    # if username == creds["username"] and password == creds["password"]:
    verify_user = db.session.query(User).filter(User.phone_number == mobile_number).first()
    if verify_user == None:
        flash("Create an account!", "danger")
        return redirect(url_for("trail_login"))
    else:
        flash("You're current account holder", "success")
        return redirect(url_for("login_2fa", mobile_number=mobile_number))


# 2FA page route
@app.route("/login/2fa/<mobile_number>")
def login_2fa(mobile_number):
    # generating random secret key for authentication
    verified_user = db.session.query(User).filter(User.phone_number == mobile_number).first()
    secret = verified_user.secret
    return render_template("login_2fa.html", secret=secret)


@app.route('/qrcode')
def qr_code_page():
    return render_template("qrcode_generator_tutorial.html")


# 2FA form route
@app.route("/login/2fa/<mobile_number>", methods=["POST"])
def login_2fa_form(mobile_number):
    # getting secret key used by user
    mobile_number = mobile_number
    secret = request.form.get("secret")
    # getting OTP provided by user
    otp = int(request.form.get("otp"))

    # verifying submitted OTP with PyOTP
    if pyotp.TOTP(secret).verify(otp):
        # inform users if OTP is valid
        flash("OTP is correct", "success")
        return redirect(url_for("display_video1"))
    else:
        # inform users if OTP is invalid
        flash("OTP is incorrect", "danger")
        return redirect(url_for("login_2fa", mobile_number=mobile_number))


# Registration page
@app.route('/trail_register')
def display_trial_register_page():
    return render_template('trial_register.html')




@app.route('/display_video_1')
def display_video1():
    return render_template('cam_video.html')


def generate_frames1():
    global value1
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # canvas = detect(gray, frame)
        # image, face =face_detector(frame)
        faces = detector(gray)
        if not ret:
            print("fail to grab frame, try again")
            break

        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            landmarks = predictor(gray, face)

            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            print('left', right_eye_ratio)
            print('right', right_eye_ratio)

            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            print('count',count)
        if img_cropped_list is not None:
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

            for i, prob in enumerate(prob_list):
                if prob > 0.98:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                    dist_list = []  # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()

                        dist_list.append(dist)
                    print('distlist of webcam', dist_list)
                    min_dist = min(dist_list)
                    max_dist = max(dist_list)  # get minumum dist value

                    min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                    # print(min_dist_idx)
                    name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                    box = boxes[
                        i]  # boundingbox(x,y,width,height) x and y coordinates reprensent the top left corner pixels
                    # other two are width and the height of the box

                    original_frame = frame.copy()  # storing copy of frame before drawing on it
                    avg = (min_dist + max_dist) / 2
                    i = int(min_dist) + 0.2

                    if min_dist < 0.90:
                        frame = cv2.putText(frame, name + ' ' + str(min_dist + 0.15), (int(box[0]), int(box[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 1, cv2.LINE_AA)

                    else:
                        frame = cv2.putText(frame, 'Unknown face', (int(box[0]), int(box[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 1, cv2.LINE_AA)
                    if blinking_ratio > 4.7:
                        global value1
                        count = count+1
                        value1 = count
                        cv2.putText(frame, "BLINKING", (int(x), int(y)), font, 3, (0, 255, 0), 1)

                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        if count >= 2:
            print('count', count)
            value1 = count
            break
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


value1 = 0


@app.route('/send_frames1')
def send_frames1():
    ''' frame = generate_frames1()
    if frame == None:
        cam.release()
        return redirect(url_for("fingerprint")) '''
    return Response(stream_with_context(generate_frames1()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/is_decoded/')
def is_decoded():
    return jsonify({'is_decoded': value1})


# Adding new user function
@app.route('/trail/add/user', methods=["POST"])
def trail_adding_user():
    value = 0
    if not request.form['user-name-input']:
        value = 1
        flash("Please enter your name", "tomato")

    if not request.form["user-age-input"]:
        value = 1
        flash("Please enter your age", "tomato")

    if not request.form["user-gender-input"]:
        value = 1
        flash("Please enter your gender", "tomato")

    if not request.form["user-address-input"]:
        value = 1
        flash("Please enter your address", "tomato")

    if not request.form["user-number-input"]:
        value = 1
        flash("Please enter your phone number", "tomato")

    if value == 0:
        mobile_number = request.form.get("user-number-input")
        verify_user = db.session.query(User).filter(User.phone_number == mobile_number).first()
        if verify_user == None:
            secret_key = pyotp.random_base32()
            new_person = User(
                name=request.form['user-name-input'],
                age=request.form['user-age-input'],
                gender=request.form['user-gender-input'],
                address=request.form['user-address-input'],
                phone_number=request.form['user-number-input'],
                secret=secret_key)
            db.session.add(new_person)
            db.session.commit()
            flash("All details are added to database successfully", "lawngreen")
        else:
            flash("Already registered")

    return redirect(url_for('display_trial_register_page'))


'''def create_session():
    session = db.sessionmaker(bind=db.engine)
    return session()
'''


# Registration page
@app.route('/register')
def display_register_page():
    return render_template('register.html')


# Invalid user page
@app.route('/invalid_user')
def invalid_user():
    return render_template('invalid_user.html')


# Adding new user function
@app.route('/add/user', methods=["POST"])
def adding_user():
    value = 0
    if not request.form['user-name-input']:
        value = 1
        flash("Please enter your name", "tomato")

    if not request.form["user-age-input"]:
        value = 1
        flash("Please enter your age", "tomato")

    if not request.form["user-gender-input"]:
        value = 1
        flash("Please enter your gender", "tomato")

    if not request.form["user-address-input"]:
        value = 1
        flash("Please enter your address", "tomato")

    if not request.form["user-number-input"]:
        value = 1
        flash("Please enter your phone number", "tomato")

    if value == 0:
        mobile_number = request.form.get("user-number-input")
        verify_user = db.session.query(User).filter(User.phone_number == mobile_number).first()
        if verify_user == None:
            secret_key = pyotp.random_base32()
            new_person = User(
                name=request.form['user-name-input'],
                age=request.form['user-age-input'],
                gender=request.form['user-gender-input'],
                address=request.form['user-address-input'],
                phone_number=request.form['user-number-input'],
                secret=secret_key)
            db.session.add(new_person)
            db.session.commit()
            flash("All details are added to database successfully", "lawngreen")
        else:
            flash("Already registered")  # New registration

    return redirect(url_for('display_register_page'))


def generate_frames():
    #camera = cv2.VideoCapture(0)
    blink = 0
    face_count = 0
    #success, frame = camera.read()
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # canvas = detect(gray, frame)
        # image, face =face_detector(frame)
        faces = detector(gray)
        face = face_extractor(frame)
        #print(face)
        #print(type(face))
        # faces = detector(gray)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            pixels = face.astype('float32')

            samples = expand_dims(pixels, axis=0)
            # prepare the face for the model, e.g. center pixels
            samples = preprocess_input(samples, )
            # im = Image.fromarray(face, 'RGB')
            # Resizing into 128x128 because we trained the model with this image size.
            # img_array = np.array(im)
            # print(img_array)
            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            # img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(samples)
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                # print('left', right_eye_ratio)
                # print('right', right_eye_ratio)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # convert one face into samples
            listp = []
            listp1 = []
            listpred = pred.tolist()
            # print(listpred)
            for i in listpred:
                listp = i
            # print(listp)
            for i in listp:
                # a = i
                a = float("{:.2f}".format(i))
                listp1.append(a)
            #print(listp1)

            maxvalue = max(listp1)
            max_index = listp1.index(maxvalue)

            #print(max_index)
            Dict = {0: 'Aditya', 1: 'Jeffrey', 2: 'Sanjay', 3: 'Saravana', 4: 'Shafeeq', 5: 'Yamini', 6: 'Bill gates',
                    7: 'Musk'}
            # printing the names and accuracy value
            name = "None matching"
            if (maxvalue >= 0.8 and max_index == 0):
                name = "Aditya" + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 1):
                name = 'Jeffrey' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 2):
                name = 'Sanjay' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 3):
                name = 'Saravana' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 4):
                name = 'Shafeeq' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 5):
                name = 'Yamini' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 6):
                name = 'Bill Gates' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            elif (maxvalue >= 0.8 and max_index == 7):
                name = 'Musk' + " " + str(maxvalue * 100) + "%"
                face_count += 1
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            if blinking_ratio > 4.6:
                blink += 1
                cv2.putText(frame, 'Blinking', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

            print('Blinkcount', blink)
            print('face_count', face_count)
            if face_count >= 5:
                break
        else:
            cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # cv2.imshow('Video', frame)
        # if blink == 5:
        # if face_count==5:
        # print("Verified successfully\n""Moving on to fingerprint recognition")
        # break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/display_video')
def display_video():
    return render_template('cam_video.html')


@app.route('/send_frames')
def send_frames():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fingerprint')
def fingerprint():
    return render_template('fingerprint.html')


@app.route('/about_us')
def about_us():
    return render_template('about_us.html')


res1 = 0.0


def generate_fingerprint():
    global res1
    frame, res1 = g1()
    print(res1)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/send_fingerprint')
def send_fingerprint():
    return Response(generate_fingerprint(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/is_matched/')
def is_matched():
    print(res1)
    return jsonify({'is_matched': res1})


# running flask server
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=90)
    # session = create_session()