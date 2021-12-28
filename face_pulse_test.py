from lib.pulse_process import findFaceGetPulse
import cv2
import numpy as np

# cameras = []

# for i in range(3):
#             camera = Camera(camera=i)  # first camera by default
#             if camera.valid or not len(cameras):
#                 cameras.append(camera)
#             else:
#                 break
# print(cameras)

w, h =0, 0
selected_cam = 0
camera = cv2.VideoCapture(selected_cam)

processor = findFaceGetPulse(bpm_limits=[50, 160],
                            data_spike_limit=2500.,
                            face_detector_smoothness=10.)

def combine(left, right):
    """Stack images horizontally.
    """
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
                col = (0,255,0)    
                ii = np.argmax(-y)
                ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                ss = ss.format(x[ii]) 
                #"%0.0f %s" % (x[ii], showmax[i])
                cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                            cv2.FONT_HERSHEY_PLAIN,2,col)
        
        try:
            pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
            i+=1
            P.append(pts)
        except ValueError:
            pass #temporary
    """ 
    #Polylines seems to have some trouble rendering multiple polys for some people
    for p in P:
        cv2.polylines(z, [p], False, (255,255,255),1)
    """
    #hack-y alternative:
    for p in P:
        for i in range(len(p)-1):
            cv2.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)    
    cv2.imshow(name,z)

def make_bpm_plot():
        """
        Creates and/or updates the data display
        """
        plotXY([[processor.times,
                 processor.samples],
                [processor.freqs,
                 processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name="plot_title",
               bg=processor.slices[0])

def test() :
    success, frame = camera.read()

    if not success : 
        print("camera not connected properly")
        exit()

    processor.frame_in = frame # masukin frame

    h, w, _c = frame.shape
    # harusnya disini ada proses apa gitu gatau
    processor.run(selected_cam)

    output_frame = processor.frame_out
    print(output_frame)
    
    make_bpm_plot()
    # show the processed frame
    return output_frame


while True :
    cv2.imshow("main", test()) 
    cv2.waitKey(20)