import numpy as np
from skimage import data, exposure
from PIL import Image
import cv2

chars = []
horz = []
vert = []


def load_csv(file):
    with open(file,'r') as filey:
        for line in filey:
            line=line.split('|')
            line[-1]=line[-1][:-1]
            for lines in line:
                lines=lines.split(',')
                lines=list(filter(None,lines))
                chars.append(lines[0])
                horz.append(lines[1:17])
                vert.append(lines[17:])



class Plate:
    h=0; w=0; angle=0; img=np.ndarray(0); letters=[np.ndarray(0)]; content='nothing'; name='';
    def __init__(self,name):
        self.name=name
        self.img=data.imread('plates/'+name+'.jpg', as_grey=True).copy()
        self.h,self.w=self.img.shape[:2]
        self.img = exposure.rescale_intensity(self.img, out_range=(0,255)).copy()
        self.img = cv2.GaussianBlur(self.img,(5,5),0)
        img2=np.zeros(shape=(self.h+2,self.w+2))
    def thresh(self):
        for row in range(self.h):
            for value in range(self.w):
                if self.img[row][value] < np.percentile(self.img,75):
                    self.img[row][value]=0
                else:
                    self.img[row][value]=255
    def invertngamma(self):
        for row in range(self.h):
            for value in range(self.w):
                self.img[row][value]=((1-self.img[row][value]/255)**2)*255
    def save(self): Image.fromarray(self.img).convert('RGB').save('plates/'+self.name+'2.jpg')
    def save_letters(self):
        self.img=cv2.convertScaleAbs(self.img)
        _,cnt,_= cv2.findContours(self.img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        i=0
        self.letters=[np.array(0) for i in range(7)]
        for countour in cnt:
            rect=cv2.minAreaRect(countour)
            box=np.int0(cv2.boxPoints(rect))
            tx=[x[0] for x in box]
            ty=[y[1] for y in box]
            boxx=min(tx),min(ty),max(tx),max(ty)
            temp_w=np.int0(max(tx)-min(tx))
            temp_h=np.int0(max(ty)-min(ty))
            if temp_w>10:
                if temp_h>10:
                    rcc=self.img[min(ty):max(ty),min(tx):max(tx)].copy()
                    for x in range(temp_h):
                        for y in range(temp_w):
                            if rcc[x][y]>0: rcc[x][y]=np.float32(255)
                            else: rcc[x][y]=np.float32(0)
                    rcc = cv2.resize( rcc, (16,16), interpolation=cv2.INTER_AREA)
                    t_horz = np.sum(rcc == 255, axis=0)
                    t_vert = np.sum(rcc == 255, axis=1)
                    t_error=10000;t_id=0;
                    for l in range(35):
                        horz[l]=[int(x) for x in horz[l]]
                        vert[l]=[int(x) for x in vert[l]]
                        t_x = t_horz - horz[l]
                        t_x = [x*x for x in t_x]
                        e_x = np.sum(t_x)
                        t_y = t_vert - vert[l]
                        t_y = [y * y for y in t_y]
                        e_y = np.sum(t_y)
                        if e_x+e_y < t_error:
                            t_error = e_x+e_y
                            t_id=l
                    print(t_error,chars[t_id])

class IMGArray:
    nparray=np.ndarray(0); h=0; w=0; ndim=0; cnt=np.ndarray(0); temp=np.array(0); bw=np.array(0); plate=(0,np.ndarray(0));address='x'
    def __init__(self,address):
        self.address=address
        self.temp = data.imread('img3/P60400'+address+'.jpg', as_grey=True).copy();
        self.bw = data.imread('img3/P60400'+address+'.jpg', as_grey=True).copy();
        self.nparray = data.imread('img3/P60400'+address+'.jpg').copy();
        self.ndim=self.nparray.ndim
        self.h,self.w=self.nparray.shape[:2]
    def save(self,name): Image.fromarray(self.nparray).convert('RGB').save('img2/'+name+'.jpg')
    def change(self):
        self.temp = cv2.GaussianBlur(self.temp,(5,5),0)
        self.temp = cv2.Sobel(self.temp, cv2.CV_8U, 1, 0, ksize=3)
        retval,self.temp=cv2.threshold(self.temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        temp_se = cv2.getStructuringElement(cv2.MORPH_RECT,(30,5))
        self.temp = cv2.morphologyEx(self.temp,cv2.MORPH_CLOSE, temp_se)
    def find_cnt(self): _,self.cnt,_ = cv2.findContours(self.temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    def color_cnt(self):
        for contour in self.cnt:
            rect=0;box=0;tx=0;ty=0;
            rect=cv2.minAreaRect(contour)
            box=np.int0(np.around(cv2.boxPoints(rect)))
            tx=[x[0] for x in box]
            ty=[y[1] for y in box]
            boxx=(min(tx),min(ty),max(tx),max(ty))
            temp_w=np.int0(max(tx)-min(tx))
            temp_h=np.int0(max(ty)-min(ty))
            if(temp_w*temp_h > 5000):
                if(temp_w>temp_h):
                    if(temp_h<50):
                        if(temp_w<250):
                            print('{} {}'.format(temp_w,temp_h))
                            cv2.drawContours(self.nparray,[box],0,(255,0,255),2)
                            input=Image.open('img3/P60400'+self.address+'.jpg', as_gray=True)
                            self.plate=input.crop(boxx)
                            self.plate.save('plates/'+self.address+'.jpg')


def init_plates(imgs):
    for value in imgs:
        newimage=IMGArray(value)
        newimage.change()
        newimage.find_cnt()
        newimage.color_cnt()
        newimage.save(value)



if __name__ == '__main__':
    numbers = ['12', '13', '15', '16', '17', '21', '27', '29', '30', '36', '37', '42', '50', '51',
               '55', '59', '63', '64', '66']
    #init_plates(numbers)
    load_csv('nn16.csv')
    for plate in numbers:
        newplate=Plate(plate)
        newplate.invertngamma()
        newplate.thresh()
        newplate.save_letters()
        newplate.save()
        break