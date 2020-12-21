import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst, GObject, GLib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import time,os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
import torch.optim as optim
import torch.nn as nn

from threading import Thread,Lock,Event

import easyocr

import pyautogui

WIDTH_old = 956
HEIGHT_old = 442
WIDTH = 1760
HEIGHT = 814
SYS_W = 100
SYS_H = 100

rect_top = 119
rect_bottom = 368

rect1_left = 262
rect1_right = 509
rect2_left = 509
rect2_right = 756
rect3_left = 756
rect3_right = 1003
rect4_left = 1003
rect4_right = 1250
rect5_left = 1250
rect5_right = 1497

status_rect_1_left = 110#['战斗中','战斗开始','结算中','摆放阶段']
status_rect_1_right = 239
status_rect_1_top = 0
status_rect_1_bottom = 37

status_rect_2_left = 165#['推荐阵容']
status_rect_2_right = 276
status_rect_2_top = 150
status_rect_2_bottom = 184

status_rect_3_left = 1480#['现有金钱']
status_rect_3_right = 1530
status_rect_3_top = 750
status_rect_3_bottom = 800

log_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
count = 0
count2 = 0

class status:
    def __init__(self):
        self.period = "未知"
        self.money = 0
        self.result = []
        self.confidence = []
        self.reader_en = easyocr.Reader(['en'])
        self.reader_ch = easyocr.Reader(['ch_sim'])
        
        self.transform = transforms.Compose([transforms.Resize(size=(224, 224)), 
                                             transforms.ToTensor()])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = torch.load("./trained_model/model_latest.pkl").eval()
        self.name_list = ["猪八戒","牛魔","裴擒虎","鲁班","上官婉儿","甄姬","孙策","陈咬金","武则天",
                          "庄周","蔡文姬","马超","嫦娥","橘右京","孙悟空","凯","张飞","诸葛亮","空",
                          "孙尚香","蒙犽","元歌","黄忠","奕星","夏侯惇","狄仁杰","明世隐","曹操",
                          "宫本武藏","钟无艳","李白","司马懿","太乙真人","吕布","姜子牙","盾山",
                          "百里守约","赵云","苏烈","孙膑","百里玄策","钟馗","不知火舞",
                          "娜可露露","大乔","周瑜","花木兰","貂蝉","典韦","刘禅","小乔","李元芳",
                          "关羽","李信","伽罗","公孙离","墨子","沈梦溪","后奕","杨玉环","刘备","老夫子"]
    
    def get_status(self, img_np):
        global count, count2, log_time
        # 选子
        img2_np = img_np[150:184,165:276,:]#h,w  推荐阵容
        img = Image.fromarray(img2_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img2_np = np.array(img)        
        result2 = self.reader_ch.recognize(img2_np.astype('uint8'))
        if result2[0][1] in ['推荐阵容'] and result2[0][-1] > 0.5:
            self.period = "选子"

            rect_left = [262,509,756,1003,1250]
            rect_right = [509,756,1003,1250,1497]
            rect_top = 119
            rect_bottom = 368

            temp = []
            for i in range(5):
                im1 = np.expand_dims(img_np[rect_top:rect_bottom,rect_left[i]:rect_right[i],:],axis=0)#hw
                tmp1 = Image.fromarray(np.squeeze(im1))
                tmp1 = self.transform(tmp1)
                temp.append(tmp1)
            im = torch.cat([tmp.unsqueeze(0) for tmp in temp],0)
            im = im.to(self.device)
            outputs = self.net(im)
            predicted = nn.functional.softmax(outputs.data,1).max(1)
            self.result = [self.name_list[index] for index in predicted.indices.cpu().numpy()]
            self.confidence = predicted.values.cpu().numpy()

            #记录置信度低的图像
            dir = "./log/"+log_time+"/piece/"
            for i in range(5):
                if self.confidence[i] < 0.8:
                    tmp1 = Image.fromarray(img_np[rect_top:rect_bottom,rect_left[i]:rect_right[i],:])
                    if not os.path.isdir(dir + self.result[i]):
                        os.makedirs(dir + self.result[i])
                    tmp1.save(dir+self.result[i]+"/"+str(count)+".jpg")
                    count += 1

            #现有金钱
            img3_np = img_np[750:800,1480:1530,:]#h,w  金钱
            img = Image.fromarray(img3_np.astype('uint8'))
            img = img.resize((img.size[0]*2,img.size[1]*2))
            img3_np = np.array(img)
            result3 = self.reader_en.recognize(img3_np.astype('uint8'))
            if result3[0][-1] > 0.3:
                self.money = result3[0][1]
                if self.money == 'o' or self.money == '':
                    self.money = '0'
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #选装备
        img4_np = img_np[30:70,700:810,:]#h,w  剩余时间
        img = Image.fromarray(img4_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img4_np = np.array(img)
        result4 = self.reader_ch.recognize(img4_np.astype('uint8'))
        if result4[0][1] in ['剩余时间'] and result4[0][-1] > 0.5:
            self.period = "选装备"
            self.result = []
            self.confidence = []
            
            rect_left = [330,560,790,1020,1250]
            rect_right = [490,720,950,1180,1410]
            for i in range(5):
                #装备选择 装备1
                img_small_np = img_np[110:150,rect_left[i]:rect_right[i],:]#h,w
                img = Image.fromarray(img_small_np.astype('uint8'))
                img = img.resize((img.size[0]*2,img.size[1]*2))
                img_small_np = np.array(img)
                r = self.reader_ch.recognize(img_small_np.astype('uint8'))
                self.result.append(r[0][1])
                self.confidence.append(r[0][-1])

                #记录置信度低的图像
                rect_left = [262, 509, 756, 1003, 1250]
                rect_right = [509, 756, 1003, 1250, 1497]
                rect_top = 119
                rect_bottom = 368
                dir = "./log/"+log_time+"/weapon/"
                for i in range(5):
                    if self.confidence[i] < 0.8:
                        tmp1 = Image.fromarray(img_np[rect_top:rect_bottom,rect_left[i]:rect_right[i],:])
                        if not os.path.isdir(dir+self.result[i]):
                            os.makedirs(dir+self.result[i])
                        tmp1.save(dir+self.result[i]+"/"+str(count2)+".jpg")
                        count2 += 1

                # dir = "./log/"+log_time+"/weapon/"
                # global count2
                # for i in range(5):
                #     tmp1 = Image.fromarray(img_np[rect_top:rect_bottom, rect_left[i]:rect_right[i], :])
                #     tmp1.save(dir + str(count2) + ".jpg")
                #     count2 += 1

            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        img1_np = img_np[0:37,110:239,:]#h,w  摆放阶段
        img = Image.fromarray(img1_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img1_np = np.array(img)
        result1 = self.reader_ch.recognize(img1_np.astype('uint8'))
        if result1[0][1] in ['战斗中','准备战斗','结算中','摆放阶段'] and result1[0][-1] > 0.5:
            self.period = result1[0][1]
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #1.组队准备
        img_small_np = img_np[650:700,900:1050,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['开始匹配'] and result[0][-1] > 0.5:
            self.period = "开始匹配"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
            
        #2.匹配成功
        img_small_np = img_np[630:680,830:920,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['确认'] and result[0][-1] > 0.5:
            self.period = "匹配成功"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
            
        #11.失败
        # img_small_np = img_np[650:700,960:1070,:]#h,w
        img_small_np = img_np[650:690,700:780,:]#h,w
        # img_small_np = img_np[330:380,820:940,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['退出'] and result[0][-1] > 0.5:
            self.period = "战斗失败"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #12.统计信息/经验更新
        img_small_np = img_np[720:780,830:930,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['继续'] and result[0][-1] > 0.5:
            self.period = "统计信息/经验更新"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #13.段位更新
        img_small_np = img_np[560:610,970:1060,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['继续'] and result[0][-1] > 0.5:
            self.period = "段位更新"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #14.经验更新，同统计信息
#         img_small_np = img_np[720:780,830:930,:]#h,w
#         img = Image.fromarray(img_small_np.astype('uint8'))
#         img = img.resize((img.size[0]*2,img.size[1]*2))
#         img_small_np = np.array(img)
#         result = self.reader_ch.recognize(img_small_np.astype('uint8'))
#         if result[0][1] in ['继续'] and result[0][-1] > 0.5:
#             self.period = "经验更新"
#             return
        
        #15.阵容回顾
        img_small_np = img_np[740:780,920:1050,:]#h,w
        # img_small_np = img_np[740:780,700:850,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['返回房间'] and result[0][-1] > 0.5:
            self.period = "阵容回顾"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        #16.名列前茅
        img_small_np = img_np[700:770,770:1000,:]#h,w
        img = Image.fromarray(img_small_np.astype('uint8'))
        img = img.resize((img.size[0]*2,img.size[1]*2))
        img_small_np = np.array(img)
        result = self.reader_ch.recognize(img_small_np.astype('uint8'))
        if result[0][1] in ['点击屏幕继续'] and result[0][-1] > 0.5:
            self.period = "名列前茅"
            return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}
        
        self.period = "未知"   
        return {"period":self.period,"money":self.money,"class":self.result,"confidence":self.confidence}

        



gimg = np.zeros((HEIGHT,WIDTH,3),dtype='uint8')
gshow = np.zeros((HEIGHT,WIDTH,3),dtype='uint8')
g_result = {"period":"未知","money":0,"class":["","","","",""],"confidence":["","","","",""]}
lock = Lock()
lock2 = Lock()
event = Event()


name_list = ["猪八戒","牛魔","裴擒虎","鲁班","上官婉儿","甄姬","孙策","陈咬金","武则天","庄周","蔡文姬","马超","嫦娥","橘右京","孙悟空","凯","张飞","诸葛亮","空","孙尚香","蒙犽","元歌","黄忠","奕星","夏侯惇","狄仁杰","明世隐","曹操","宫本武藏","钟无艳","李白","司马懿","太乙真人","吕布","姜子牙","盾山","百里守约","赵云","苏烈","孙膑","百里玄策","钟馗","后奕","不知火舞","娜可露露","大乔","周瑜","花木兰","貂蝉","典韦","刘禅","小乔","李元芳","关羽","李信","伽罗","公孙离","墨子","沈梦溪","杨玉环","刘备","老夫子"]


def cb_busmessage_state_change(bus,message,pipeline):
    if message.src == pipeline:
        oldstate,newstate,pending = message.parse_state_changed()
        old = Gst.Element.state_get_name(oldstate)
        new = Gst.Element.state_get_name(newstate)
        print("pipeline state changed from ",old," to ",new)

from fractions import Fraction
pts = 0
duration = 10**9 / Fraction(30)
def cb_appsrc(appsrc, b):#显示
    #print("hi form cb_appsrc")
    global name_list
    global gimg,lock2,g_result
    global pts,duration

    lock2.acquire()
    #array = np.random.randint(low=0,high=255,size=(1080,1920,3),dtype="uint8")
    array = gshow.copy()
    result = g_result.copy()
    lock2.release()
    
           
    im = Image.fromarray(array)
    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",30)
    if result['period'] in ['选子',"选装备"]: 
        #棋子种类
        draw.text((312,500),result['class'][0],font=fnt, fill=(255,0,0))#w,h
        draw.text((580,500),result['class'][1],font=fnt, fill=(255,0,0))#w,h
        draw.text((846,500),result['class'][2],font=fnt, fill=(255,0,0))#w,h
        draw.text((1113,500),result['class'][3],font=fnt, fill=(255,0,0))#w,h
        draw.text((1380,500),result['class'][4],font=fnt, fill=(255,0,0))#w,h
        #置信度
        draw.text((312,550),str(round(result['confidence'][0],3)),font=fnt, fill=(255,0,0))#w,h
        draw.text((580,550),str(round(result['confidence'][1],3)),font=fnt, fill=(255,0,0))#w,h
        draw.text((846,550),str(round(result['confidence'][2],3)),font=fnt, fill=(255,0,0))#w,h
        draw.text((1113,550),str(round(result['confidence'][3],3)),font=fnt, fill=(255,0,0))#w,h
        draw.text((1380,550),str(round(result['confidence'][4],3)),font=fnt, fill=(255,0,0))#w,h

        #棋子框横线
        draw.line((rect1_left, rect_top, rect5_right, rect_top), 'red',5)#w1,h1,w2,h2
        draw.line((rect1_left, rect_bottom, rect5_right, rect_bottom), 'red',5)#w1,h1,w2,h2
        #棋子框竖线
        draw.line((rect1_left, rect_top, rect1_left, rect_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((rect1_right, rect_top, rect1_right, rect_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((rect3_left, rect_top, rect3_left, rect_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((rect3_right, rect_top, rect3_right, rect_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((rect5_left, rect_top, rect5_left, rect_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((rect5_right, rect_top, rect5_right, rect_bottom), 'red',5)#w1,h1,w2,h2
        
        if g_result['period'] in ['选子']:
            #推荐阵容
            draw.line((status_rect_2_left, status_rect_2_top, status_rect_2_left, status_rect_2_bottom), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_2_right, status_rect_2_top, status_rect_2_right, status_rect_2_bottom), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_2_left, status_rect_2_top, status_rect_2_right, status_rect_2_top), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_2_left, status_rect_2_bottom, status_rect_2_right, status_rect_2_bottom), 'red',5)#w1,h1,w2,h2
            draw.text((status_rect_2_left,status_rect_2_bottom+20),"推荐阵容",font=fnt, fill=(255,0,0))#w,h

            #现有金钱框
            draw.line((status_rect_3_left, status_rect_3_top, status_rect_3_left, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_3_right, status_rect_3_top, status_rect_3_right, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_3_left, status_rect_3_top, status_rect_3_right, status_rect_3_top), 'red',5)#w1,h1,w2,h2
            draw.line((status_rect_3_left, status_rect_3_bottom, status_rect_3_right, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
            draw.text((status_rect_3_right+20,int((status_rect_3_top+status_rect_3_bottom)/2)),str(result['money']),font=fnt, fill=(255,0,0))#w,h
    #        draw.text((310,600),name_list[result[0][0]],font=fnt, fill=(255,0,0))
    #        draw.text((635,600),name_list[result[0][1]],font=fnt, fill=(255,0,0))
    #        draw.text((960,600),name_list[result[0][2]],font=fnt, fill=(255,0,0))
    #        draw.text((1285,600),name_list[result[0][3]],font=fnt, fill=(255,0,0))
    #        draw.text((1610,600),name_list[result[0][4]],fon奕t=fnt, fill=(255,0,0))
    #
    #        draw.text((310,550),str(result[1][0]),font=fnt, fill=(255,0,0))
    #        draw.text((635,550),str(result[1][1]),font=fnt, fill=(255,0,0))
    #        draw.text((960,550),str(result[1][2]),font=fnt, fill=(255,0,0))
    #        draw.text((1285,550),str(result[1][3]),font=fnt, fill=(255,0,0))
    #        draw.text((1610,550),str(result[1][4]),font=fnt, fill=(255,0,0))

    #        draw.line((148, 160, 1773,160), 'red',10)#w,h
    #        draw.line((148, 486, 1773,486), 'red',10)#w,h
    #
    #        draw.line((148, 160, 148,486), 'red',10)#w,h
    #        draw.line((473, 160, 473,486), 'red',10)#w,h
    #        draw.line((798, 160,798 ,486), 'red',10)#w,h
    #        draw.line((1123, 160,1123 ,486), 'red',10)#w,h
    #        draw.line((1448, 160,1448 ,486), 'red',10)#w,h
    #        draw.line((1773, 160,1773 ,486), 'red',10)#w,h
    elif result['period'] in ['战斗中','战斗开始','结算中','摆放阶段']:
        #战斗中
        draw.line((status_rect_1_left, status_rect_1_top, status_rect_1_left, status_rect_1_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_1_right, status_rect_1_top, status_rect_1_right, status_rect_1_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_1_left, status_rect_1_top, status_rect_1_right, status_rect_1_top), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_1_left, status_rect_1_bottom, status_rect_1_right, status_rect_1_bottom), 'red',5)#w1,h1,w2,h2
        draw.text((status_rect_1_left,int((status_rect_1_top+status_rect_1_bottom)/2)+20),result['period'],font=fnt, fill=(255,0,0))#w,h
        
        #现有金钱框
        draw.line((status_rect_3_left, status_rect_3_top, status_rect_3_left, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_3_right, status_rect_3_top, status_rect_3_right, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_3_left, status_rect_3_top, status_rect_3_right, status_rect_3_top), 'red',5)#w1,h1,w2,h2
        draw.line((status_rect_3_left, status_rect_3_bottom, status_rect_3_right, status_rect_3_bottom), 'red',5)#w1,h1,w2,h2
        draw.text((status_rect_3_right+20,int((status_rect_3_top+status_rect_3_bottom)/2)),str(result['money']),font=fnt, fill=(255,0,0))#w,h
        
    draw.text((1500,600),"status:"+result['period'],font=fnt, fill=(255,0,0))
    draw.text((1500,650),"money:"+str(result['money']),font=fnt, fill=(255,0,0))
    array = np.asarray(im)
    
    gst_buffer = Gst.Buffer.new_wrapped(array.tobytes())
    pts += duration
    gst_buffer.pts = pts
    gst_buffer.duration = duration
    ret = appsrc.emit("push-buffer",gst_buffer)

    #print("hi from cb_appsrc end")
    return Gst.FlowReturn.OK

def cb_appsink(appsink):
    global gimg,lock,event
    #print("hi from cb_appsink")
    sample = appsink.emit("pull-sample")
    gst_buffer = sample.get_buffer()  # Gst.Buffer    
    array = np.ndarray(shape=(HEIGHT,WIDTH,3),buffer=gst_buffer.extract_dup(0,gst_buffer.get_size()),dtype='uint8')
    #print(gst_buffer.get_size())

#    im1 = np.expand_dims(array[160:486,148:473,:],axis=0)
#    tmp1 = Image.fromarray(np.squeeze(im1))
#    tmp1 = transform(tmp1)
#    #tmp.save("1.jpg")
#    im2 = np.expand_dims(array[160:486,473:798,:],axis=0)
#    tmp2 = Image.fromarray(np.squeeze(im2))
#    tmp2 = transform(tmp2)
#    #tmp.save("2.jpg")
#    im3 = np.expand_dims(array[160:486,798:1123,:],axis=0)
#    tmp3 = Image.fromarray(np.squeeze(im3))
#    tmp3 = transform(tmp3)
#    #tmp.save("3.jpg")
#    im4 = np.expand_dims(array[160:486,1123:1448,:],axis=0)
#    tmp4 = Image.fromarray(np.squeeze(im4))
#    tmp4 = transform(tmp4)
#    #tmp.save("4.jpg")
#    im5 = np.expand_dims(array[160:486,1448:1773,:],axis=0)
#    tmp5 = Image.fromarray(np.squeeze(im5))
#    tmp5 = transform(tmp5)
#    #tmp.save("5.jpg")

    lock.acquire()
    gimg = array.copy()
    lock.release()
    
    event.set()
    #print("hi from cb_appsink end")
    return Gst.FlowReturn.OK

def gst_init():
    Gst.init([])
    #pipeline = Gst.parse_launch("filesrc location=a.flv ! flvdemux ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=video/x-raw,format=RGB,framerate=30/1 ! appsink emit-signals=True")
    #pipeline = Gst.parse_launch("souphttpsrc location=http://tx2play1.douyucdn.cn/live/7994399rCnl7guad.flv?uuid= ! flvdemux ! h264parse ! avdec_h264 ! videoconvert ! capsfilter caps=video/x-raw,format=RGB ! videorate ! capsfilter caps=video/x-raw,format=RGB,framerate=1/1 ! appsink emit-signals=True")
    #pipeline = Gst.parse_launch("videotestsrc num-buffers=100 ! videoconvert ! video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! appsink emit-signals=True")
    #pipeline = Gst.parse_launch("souphttpsrc location=http://tx2play1.douyucdn.cn/live/1984839rdqYitoI8.flv?uuid= ! flvdemux ! h264parse ! flvmux ! rtmpsink location='rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_276236640_89465078&key=ebee02ef9f8800960276c861ce05dacb&schedule=rtmp'")
    pipeline = Gst.parse_launch("ximagesrc xname=mi9 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width={},height={},framerate=30/1 ! appsink emit-signals=True".format(WIDTH,HEIGHT))

    #pipeline = Gst.parse_launch("appsrc is-live=True do-timestamp=True emit-signals=True block=True stream-type=0 format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! videoconvert ! x264enc ! h264parse ! flvmux ! filesink location=appsrc.flv")
    #pipeline = Gst.parse_launch("appsrc is-live=True do-timestamp=True emit-signals=True block=True stream-type=0 format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! videoconvert ! appsink")
    #pipeline2 = Gst.parse_launch("appsrc is-live=True do-timestamp=True emit-signals=True block=True stream-type=0 format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1 ! videoconvert ! x264enc ! h264parse ! flvmux ! rtmpsink location='rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_276236640_89465078&key=ebee02ef9f8800960276c861ce05dacb&schedule=rtmp'")
    # pipeline2 = Gst.parse_launch("appsrc is-live=True do-timestamp=True emit-signals=True block=True stream-type=0 format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width={},height={},framerate=30/1 ! videoconvert ! videoscale ! autovideosink".format(WIDTH,HEIGHT))    # video/x-raw,format=RGB,width=880,height=400,framerate=30/1 !
    pipeline2 = Gst.parse_launch("appsrc is-live=True do-timestamp=True emit-signals=True block=True stream-type=0 format=GST_FORMAT_TIME caps=video/x-raw,format=RGB,width={},height={},framerate=30/1 ! videoconvert ! videoscale ! x264enc ! h264parse ! flvmux ! rtmpsink location='rtmp://live-push.bilivideo.com/live-bvc/?streamname=live_276236640_89465078&key=ebee02ef9f8800960276c861ce05dacb&schedule=rtmp'".format(WIDTH,HEIGHT))    # video/x-raw,format=RGB,width=880,height=400,framerate=30/1 !

    appsink = pipeline.get_by_name("appsink0")
    appsink.connect("new-sample",cb_appsink)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::state-changed",cb_busmessage_state_change,pipeline)
    
    appsrc = pipeline2.get_by_name("appsrc0")
    appsrc.connect("need-data",cb_appsrc)
    bus2 = pipeline2.get_bus()
    bus2.add_signal_watch()
    bus2.connect("message::state-changed",cb_busmessage_state_change,pipeline2)

    return pipeline, pipeline2

def action_init():
    def is_target(result):
        target_name_list = ["马超","张飞","诸葛亮","元歌","黄忠","赵云","刘禅","关羽","刘备"]
        if result in target_name_list:
            return True
        else:
            return False
        
    def action():
        global lock2,g_result
        button_location = [[(rect1_left+rect1_right)/2,(rect_top+rect_bottom)/2],[(rect2_left+rect2_right)/2,(rect_top+rect_bottom)/2],[(rect3_left+rect3_right)/2,(rect_top+rect_bottom)/2],[(rect4_left+rect4_right)/2,(rect_top+rect_bottom)/2],[(rect5_left+rect5_right)/2,(rect_top+rect_bottom)/2],]#w,h
        while True:
            #print("hi from action")
            button = [0,0,0,0,0]
            button2 = [0]
            lock2.acquire()
            if g_result['period'] in ['选子']:
                for i in range(5):
                    if is_target(g_result['class'][i]):
                        button[i] = 1
                
                button2[0] = 1
                if int(g_result['money']) > 5:
                    button2_location = (1540,340)
                else:
                    button2_location = (1500,730)
            elif g_result['period'] in ['选装备']:
                button[0] = 1
                button[1] = 1

                button2[0] = 1
                button2_location = (1500, 730)
            elif g_result['period'] in ['开始匹配']:
                button2_location = (975,675)#w,h
                button2[0] = 1
            elif g_result['period'] in ['匹配成功']:
                button2_location = (875,655)#w,h
                button2[0] = 1   
            elif g_result['period'] in ['战斗失败']:
                button2_location = (740,670)#w,h
                button2[0] = 1    
            elif g_result['period'] in ['统计信息/经验更新']:
                button2_location = (880,750)#w,h
                button2[0] = 1
            elif g_result['period'] in ['段位更新']:
                button2_location = (985,585)#w,h
                button2[0] = 1
            elif g_result['period'] in ['阵容回顾']:
                button2_location = (985,760)#w,h
                button2[0] = 1
            elif g_result['period'] in ['名列前茅']:
                button2_location = (985,760)#w,h
                button2[0] = 1
            lock2.release()

            for i in range(5):
                if button[i] == 1:
                    pyautogui.moveTo((button_location[i][0]+SYS_W,button_location[i][1]+SYS_H))
                    time.sleep(0.3)
                    pyautogui.click()
            
            if button2[0] == 1:
                pyautogui.moveTo((button2_location[0]+SYS_W,button2_location[1]+SYS_H))
                time.sleep(0.3)
                pyautogui.click()
            
            time.sleep(1)
            
            
    t = Thread(target=action, name='action')
    t.start()
    
def detect_init():
    def detect():
        global gimg,gshow,lock,lock2,event,g_result
        s = status()
        while(1):
            event.wait()
                    
            lock.acquire()
            array = gimg.copy()
            lock.release()
            
            result = s.get_status(array)
            
            lock2.acquire()
            gshow = array.copy()
            g_result = result.copy()
            lock2.release()
            
            
        
        
    t = Thread(target=detect, name='detect')
    t.start()

if __name__ == "__main__":
    action_init()
    detect_init()
    pipeline, pipeline2 = gst_init()
    ret = pipeline.set_state(Gst.State.PLAYING)
    ret = pipeline2.set_state(Gst.State.PLAYING)
    GLib.MainLoop().run()
