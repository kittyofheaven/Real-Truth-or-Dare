from flask import Flask, render_template, Response
from lib.pulse_process import findFaceGetPulse
from voice_recog import get_audio
import numpy as np
import cv2
from time import sleep
from flask_socketio import SocketIO, send
from voice_recog import get_audio, audio_callback

""" TO-DO
[x] config voice
[ ] get the wpm
[ ] config video process
[ ] extract pulse & measure
[ ] get bpm
[ ] replicate polygraph
[ ] style front-end
"""


#PARAMETERS#

############

app = Flask(__name__)
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')