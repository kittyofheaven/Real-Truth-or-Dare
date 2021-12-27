from speech_recognition import Recognizer, Microphone, UnknownValueError, RequestError
import pickle
import wave
import contextlib

def model_predict(sentence:str) :
    model = pickle.load(open('lib/model_0.pkl', 'rb'))
    predict = model.predict([sentence])
    return predict[0]

def word_per_minute(string:str, sec:int) : 
    string = string.split(" ")
    wpm = len(string) / (sec / 60) 
    wpm = round(wpm)
    return wpm

r = Recognizer()
mic = Microphone()

def audio_callback(recognizer, audio) : 

    try:
        fname = 'lib/microphone-results.wav'
        with open(fname, "wb") as f:
            f.write(audio.get_wav_data())
    
        with contextlib.closing(wave.open(fname,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            time_taken = round(frames / float(rate))
                # use relative path    
        # time_taken = int(wav.info.length)
        text = recognizer.recognize_google(audio)
        sentiment = model_predict(text)
        responses = {"text" :  text, "time_taken" : time_taken, "wpm" : word_per_minute(text, time_taken), "sentiment" : sentiment}
    
    except RequestError as exc :
        responses = {"text" : str(exc), "sentiment" : 400}

    except UnknownValueError:
        responses = {"text" : "couldnt recognize ur voice", "sentiment" : 400}
    
    audio_callback.response = responses 
    return responses

def print_response(response) : 
    print(response)


def get_audio() :
    with mic as source:
        r.adjust_for_ambient_noise(mic)
        # start = time.time()
        # print('Speak Anything : ')

    get_audio.stop_audio = r.listen_in_background(mic, audio_callback)

# get_audio()

# time.sleep(1000) # still listening even though the main thread is doing other things

# get_audio.stop_audio(wait_for_stop=False) # for stopping audio listen
