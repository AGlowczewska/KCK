import numpy as np
from skimage import data, io, exposure, filters
from PIL import Image
import cv2

class IMGArray:
    nparray=np.ndarray(0); h=0; w=0; ndim=0; cnt=np.ndarray(0); temp=np.array(0)
    def __init__(self,address):
        self.temp = data.imread('img/'+address+'.jpg', as_grey=True).copy();
        self.nparray = data.imread('img/'+address+'.jpg').copy();
        self.ndim=self.nparray.ndim
        self.h,self.w=self.nparray.shape[:2]
    def convertion(self): self.temp = exposure.rescale_intensity(self.temp, out_range=(0,255))
    def save(self,name): Image.fromarray(self.nparray).convert('RGB').save('img2/'+name+'.jpg')
    def gamma(self,n):
        for row in range(self.h):
            for value in range(self.w):
                self.temp[row][value]=self.temp[row][value]**n
    def negate(self):
        for row in range(self.h):
            for value in range(self.w):
                self.temp[row][value] = 1 - self.temp[row][value]
    def change(self):
        self.temp = cv2.GaussianBlur(self.temp,(5,5),0)
        self.temp = cv2.Sobel(self.temp, cv2.CV_8U, 1, 0, ksize=3)
        retval,self.temp=cv2.threshold(self.temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        temp_se = cv2.getStructuringElement(cv2.MORPH_RECT,(50,2))
        self.temp = cv2.morphologyEx(self.temp,cv2.MORPH_CLOSE, temp_se)
    def find_cnt(self): _,self.cnt,_ = cv2.findContours(self.temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    def color_cnt(self):
        for contour in self.cnt:
            rect=cv2.minAreaRect(contour)
            box=np.int0(np.around(cv2.boxPoints(rect)))
            temp_w=rect[1][0]
            temp_h=rect[1][1]
            if((temp_w>0) & (temp_h>0)):
                if((((temp_w/temp_h) <6)&(temp_w>temp_h))|(((temp_h/temp_w) <6)&(temp_w<temp_h))):
                    if((temp_w*temp_h > 5000) & (temp_h*temp_w < 50000)):
                        cv2.drawContours(self.nparray,[box],0,(255,0,255),2)




if __name__ == '__main__':
    numbers = ['01', '02', '03','04','05','06','07','08','09']
    for value in numbers:
        newimage=IMGArray(value)
        #newimage.negate()
        #newimage.gamma(2)
        #newimage.convertion()
        newimage.change()
        newimage.find_cnt()
        newimage.color_cnt()
        newimage.save(value)