from flask import Flask, render_template, Response
from lib.pulse_process import findFaceGetPulse
from voice_recog import get_audio
import numpy as np
import cv2
from time import sleep
from flask_socketio import SocketIO, send
from voice_recog import get_audio, audio_callback