import numpy as np
import time
import cv2
import pylab
import os
import sys

"""TO-DO
[x] find face
[x] get the pulse
[x] return the bpm
"""

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained


    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        # INI YG BUAT FOREHEAD GERAK
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                            scaleFactor=1.3,
                                                            minNeighbors=4,
                                                            minSize=(
                                                                50, 50),
                                                            flags=cv2.CASCADE_SCALE_IMAGE))
        
        if len(detected) > 0:
            
            detected.sort(key=lambda a: a[-1] * a[-2])
            

            if self.shift(detected[-1]) > 10:
                self.face_rect = detected[-1]
        #  tapi masalahnya klo ada kode diatas kenapa mereka gamau detect

        # pas setelah user pencet s bakal ngelock dan forehead locknya yang dibawah ini
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1, col=(0, 0, 255))
        # x, y, w, h = forehead1

        vals = self.get_subface_means(forehead1)

        self.data_buffer.append(vals) # whats happening in here pas gaada yg forehead gerak vals nya ke append tapi begitu ada dia cuma ke append 1 kali
        
        buffer_len = len(self.data_buffer)
        # print(vals, buffer_len, self.buffer_size, self.data_buffer)

        if len(detected) > 0 :

            if buffer_len > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                buffer_len = self.buffer_size

            processed = np.array(self.data_buffer)
            self.samples = processed
            
            if buffer_len > 10:
                self.output_dim = processed.shape[0]
                
                self.fps = float(buffer_len) / (self.times[-1] - self.times[0])
                even_times = np.linspace(self.times[0], self.times[-1], buffer_len)
                interpolated = np.interp(even_times, self.times, processed)
                interpolated = np.hamming(buffer_len) * interpolated
                interpolated = interpolated - np.mean(interpolated)
                raw = np.fft.rfft(interpolated)
                phase = np.angle(raw)
                self.fft = np.abs(raw)
                self.freqs = float(self.fps) / buffer_len * np.arange(buffer_len / 2 + 1)

                freqs = 60. * self.freqs
                idx = np.where((freqs > 50) & (freqs < 180))

                pruned = self.fft[idx]
                phase = phase[idx]

                pfreq = freqs[idx]
                self.freqs = pfreq
                self.fft = pruned
                idx2 = np.argmax(pruned)

                t = (np.sin(phase[idx2]) + 1.) / 2.
                t = 0.9 * t + 0.1
                alpha = t
                beta = 1 - t

                self.bpm = self.freqs[idx2]
                self.idx += 1

                x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
                r = alpha * self.frame_in[y:y + h, x:x + w, 0]
                g = alpha * \
                    self.frame_in[y:y + h, x:x + w, 1] + \
                    beta * self.gray[y:y + h, x:x + w]
                b = alpha * self.frame_in[y:y + h, x:x + w, 2]
                self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                            g,
                                                            b])
                x1, y1, w1, h1 = self.face_rect
                self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
                text = "(estimate: %0.1f bpm, fps %0.0f )" % (self.bpm, self.fps)
                cv2.putText(self.frame_out, text,(int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 1, 255), 2)
