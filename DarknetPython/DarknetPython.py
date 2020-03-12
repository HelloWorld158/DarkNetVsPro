import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
import numpy as np
import matplotlib.image as imgread
import darknet as dn
import pdb
import glob
from PIL import Image,ImageDraw,ImageFont
import platform
import time
dn.set_gpu(0)
a=platform.system()
if(a=='Windows'):
    path="..\\..\\Detection"
    sep="\\"
elif(a=='Linux'):
    path="/valTestData"
    sep="/"
imagedir=path
imagedir=os.path.abspath(imagedir)+sep
imagelist=[file for file in glob.glob(os.path.join(imagedir,'*.jpg'))]
print(imagedir)
net = dn.load_net((imagedir+"yolov3.cfg").encode("UTF-8"), (imagedir+"yolov3.weights").encode("UTF-8"), 0)
if(a=='Windows'):
    coconame="coco.data"
elif(a=='Linux'):
    coconame="coco-linux.data"
filecoco=imagedir+coconame
coconame=filecoco
meta = dn.load_meta(coconame.encode("UTF-8"))
timeCount=0
iCount=0
alltimes=time.perf_counter()
for i in imagelist:
    k=i.encode('UTF-8')
    start=time.perf_counter()
    r = dn.detect(net, meta, k)
    end=time.perf_counter()
    timeCount+=end-start
    img=Image.open(k)
    idraw=ImageDraw.Draw(img)
    for j in r:
        idraw.text((j[2][0]-j[2][2]/2,j[2][1]-j[2][3]/2),j[0])
        idraw.rectangle((j[2][0]-j[2][2]/2,j[2][1]-j[2][3]/2,j[2][0]+j[2][2]/2,j[2][1]+j[2][3]/2),width=2)
    t=np.asarray(img)
    (filepath,tempfilename) = os.path.split(i);
    (shortname,extension) = os.path.splitext(tempfilename);
    shortname+="pdict"
    fullpath=filepath+sep+shortname+extension
    if(a=='Windows'):
        fullpath=fullpath.encode("UTF-8")
    elif(a=="Linux"):
        filepath="/valRes"
        fullpath=filepath+sep+shortname+extension
    #print(fullpath)
    imgread.imsave(fullpath,t)
    print("finish:",iCount,"/",len(imagelist),end="\r",flush=True)
    iCount+=1
print('\nAll Extract Time:',time.perf_counter()-alltimes)
print('time consume:',timeCount);


    

