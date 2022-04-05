from keras.models import load_model
import sys
sys.path.append("C:\\Users\\Luis Puente\\OneDrive\\Escritorio\\Art_Seis\\OpendTect-External-Attributes-master\\Python_3")
from unet3 import *



# set gaussian weights in the overlap boundaries
def getMask(os):
    n1, n2, n3 = 128, 128, 128
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)
    for k1 in range(os):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k1]
                sc[n1-k1-1][k2][k3]=sp[k1]
    for k1 in range(n1):
        for k2 in range(os):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k2]
                sc[k1][n3-k2-1][k3]=sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(os):
                sc[k1][k2][k3]=sp[k3]
                sc[k1][k2][n3-k3-1]=sp[k3]
    return sc




def redneu(gx):
    n1, n2, n3 = 128, 128, 128
    m1,m2,m3 = gx.shape

    json_file = open('C:\\Users\\Luis Puente\\OneDrive\\Escritorio\\faultSeg-master\\model\\model3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:\\Users\\Luis Puente\\OneDrive\\Escritorio\\faultSeg-master\\model\\pretrained_model.hdf5")
    os = 12 #overlap width
    c1 = np.round((m1+os)/(n1-os)+0.5)
    c2 = np.round((m2+os)/(n2-os)+0.5)
    c3 = np.round((m3+os)/(n3-os)+0.5)
    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)
    p1 = (n1-os)*c1+os
    p2 = (n2-os)*c2+os
    p3 = (n3-os)*c3+os
    gx = np.reshape(gx,(m1,m2,m3))
    gp = np.zeros((p1,p2,p3),dtype=np.single)
    gy = np.zeros((p1,p2,p3),dtype=np.single)
    mk = np.zeros((p1,p2,p3),dtype=np.single)
    gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
    gp[0:m1,0:m2,0:m3]=gx
    sc = getMask(os)
    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                b1 = k1*n1-k1*os
                e1 = b1+n1
                b2 = k2*n2-k2*os
                e2 = b2+n2
                b3 = k3*n3-k3*os
                e3 = b3+n3
                gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
                gs = gs-np.min(gs)
                gs = gs/np.max(gs)
                gs = gs*255
                Y = loaded_model.predict(gs,verbose=0)
                Y = np.array(Y)
                gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
                mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
    gy = gy/mk
    gy = gy[0:m1,0:m2,0:m3]
    return gy
