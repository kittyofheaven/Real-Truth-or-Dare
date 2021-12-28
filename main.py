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
[x] send it to frontend # socket io (?)
[x] get the wpm
[x] config video process
[x] extract pulse & measure
[x] get bpm
[x] replicate polygraph
[ ] style front-end
[x] add participant parameter
"""


#PARAMETERS#
selected_cam = 0
num_of_player = 2
############

if num_of_player == 1 :
    num_of_question = 2
else :
    num_of_question = num_of_player

app = Flask(__name__)
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")
processor = findFaceGetPulse(bpm_limits=[50, 160],
                            data_spike_limit=2500.,
                            face_detector_smoothness=10.)


def combine(left, right):
    # combine images horizontally.
    
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    hoff = left.shape[0]
    
    shape = list(left.shape)
    shape[0] = h
    shape[1] = w
    
    comb = np.zeros(tuple(shape),left.dtype)
    
    # left will be on left, aligned top, with right on right
    comb[:left.shape[0],:left.shape[1]] = left
    comb[:right.shape[0],left.shape[1]:] = right
    
    return comb  

def plotXY(data,size = (280,640),margin = 25,name = "data",labels=[], skip = [],
           showmax = [], bg = None,label_ndigits = [], showmax_digits=[]):
    for x,y in data:
        if len(x) < 2 or len(y) < 2:
            return
    
    n_plots = len(data)
    w = float(size[1])
    h = size[0]/float(n_plots)
    
    z = np.zeros((size[0],size[1],3))
    
    if isinstance(bg,np.ndarray):
        wd = int(bg.shape[1]/bg.shape[0]*h )
        bg = cv2.resize(bg,(wd,int(h)))
        if len(bg.shape) == 3:
            r = combine(bg[:,:,0],z[:,:,0])
            g = combine(bg[:,:,1],z[:,:,1])
            b = combine(bg[:,:,2],z[:,:,2])
        else:
            r = combine(bg,z[:,:,0])
            g = combine(bg,z[:,:,1])
            b = combine(bg,z[:,:,2])
        z = cv2.merge([r,g,b])[:,:-wd,]    
    
    i = 0
    P = []
    count_bpm = 0 
    for x,y in data:
        x = np.array(x)
        y = -np.array(y)
        
        xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
        yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h
        mx = max(yy)
        if labels:
            if labels[i]:
                for ii in range(len(x)):
                    if ii%skip[i] == 0:
                        col = (255,255,255)
                        ss = '{0:.%sf}' % label_ndigits[i]
                        ss = ss.format(x[ii]) 
                        cv2.putText(z,ss,(int(xx[ii]),int((i+1)*h)),
                                    cv2.FONT_HERSHEY_PLAIN,1,col)           
        if showmax:
            if showmax[i]:
                col = (0,0,255)    
                ii = np.argmax(-y)
                ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                ss = ss.format(x[ii]) 
                count_bpm = float(ss.split(" ")[0])
                #"%0.0f %s" % (x[ii], showmax[i])
                cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                            cv2.FONT_HERSHEY_PLAIN,2,col)
        
        try:
            pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
            i+=1
            P.append(pts)
        except ValueError:
            pass 
        
    for p in P:
        for i in range(len(p)-1):
            cv2.line(z,tuple(p[i]),tuple(p[i+1]), (0,255,0),1)    

    # coloured = resized.copy()
    # coloured[mask == 255] = (255, 255, 255)

    ret, jpeg = cv2.imencode('.jpg', z)

    # if jpeg == None :
    #     print("NONEEEEE INI WOIIII")
    
    # print(jpeg)
    # print("NONEEEEE INI WOIIII")
    return {"jpeg" : jpeg.tobytes(), "bpm" : count_bpm}

def get_bpm() : 
    responses = plotXY([[processor.times,
                processor.samples],
                [processor.freqs,
                processor.fft]],
                labels=[False, True],
                showmax=[False, "bpm"],
                label_ndigits=[0, 0],
                showmax_digits=[0, 1],
                skip=[3, 3],
                name="plot_title",
        )
    return responses["bpm"]

def make_bpm_plot():
    """
    Creates and/or updates the data display
    """
    responses = plotXY([[processor.times,
                processor.samples],
                [processor.freqs,
                processor.fft]],
                labels=[False, True],
                showmax=[False, "bpm"],
                label_ndigits=[0, 0],
                showmax_digits=[0, 1],
                skip=[3, 3],
                name="plot_title",
        )
    return responses["jpeg"]
        # cv2.imshow("main", frame) 
        # cv2.waitKey(20)
    
        # yield (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                

# CAMERA START #
w, h =0, 0
camera = cv2.VideoCapture(selected_cam)



w, h =0, 0

def get_frame():

    success, frame = camera.read()
    processor.frame_in = frame # masukin frame

    h, w, _c = frame.shape
    # harusnya disini ada proses apa gitu gatau
    processor.run(selected_cam)

    output_frame = processor.frame_out
    ret, jpeg = cv2.imencode('.jpg', output_frame)
    return jpeg.tobytes()

def gen(video_func):
    while True:
        frame = video_func()
        # cv2.imshow("main", frame) 
        # cv2.waitKey(20)

        if frame == None :
            img_1 = np.zeros([100,100,1],dtype=np.uint8)
            img_1.fill(255)
            frame = img_1.tobytes() # THIS IS A FEATURE, THIS IS NOT A BUG OKAY !!!

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# CAMERA END #

def average_list(lst):
    return sum(lst) / len(lst)

# ROUTE START #

@socketio.on('message')
def main_handle_voice(msg) :
    send(msg, broadcast=True)
    print(msg)
    
    average_bpm_list = []
    average_wpm_list = []

    get_audio()
    audio_callback.response = None
    already = None
    while True :
        if audio_callback.response == None :
            sleep(0.3)

        elif audio_callback.response == already :
            sleep(0.3)

        else :
            already = audio_callback.response 
            print(audio_callback.response) # disini nanti masukin yang buat flask itu (yg append ke htmlnya)

            if audio_callback.response['sentiment'] == 400:
                send_response = "error recognizing ur voice"

            else : 
                if audio_callback.response['sentiment'] == 0 :
                    sentiment = "negative"
            
                else :
                    sentiment = "neutral"
                
                if len(average_bpm_list) < num_of_question :
                    bpm = get_bpm()
                    average_bpm_list.append(bpm)
                    average_wpm_list.append(audio_callback.response['wpm'])
                    send_response = "'"+audio_callback.response['text'] + ",' \nWPM : " + str(audio_callback.response['wpm']) + ", \nBPM :" + str(bpm) + ", \nSentiment : " + sentiment + ", \nAnomaly : " + "calculating player " +str(len(average_bpm_list))

                else : 
                    average_bpm = average_list(average_bpm_list)
                    average_wpm = average_list(average_wpm_list)
                    bpm = get_bpm()
                    wpm = audio_callback.response['wpm']
                    if abs(average_bpm - bpm) > 5 :
                        if sentiment == "neutral" : 
                            anomaly = (abs(round(average_wpm) - wpm)  + abs(round(average_bpm - bpm))) / 2 
                        elif sentiment == "negative" : 
                            anomaly = (abs(round(average_wpm) - wpm)  + abs(round(average_bpm - bpm))) * 2
                    else : 
                        anomaly = 0
                    send_response = "'"+audio_callback.response['text'] + ",' \nWPM : " +  str(wpm) + ", \nBPM :" + str(bpm) + ", \nSentiment : " + sentiment + ", \nAnomaly : " + str(anomaly)
            send(send_response, broadcast=True)


@app.route('/')
def index():
    return render_template('index.html') # disini nanti masukin beberapa yang dari voice recog itu

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot_feed')
def plot_feed():
    # print(get_bpm())
    return Response(gen(make_bpm_plot),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bpm')
def bpm():
    return str(get_bpm())
# ROUTE END #

if __name__ == '__main__':
    socketio.run(app)
    # app.run(host='0.0.0.0', debug=True)