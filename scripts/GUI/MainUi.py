# coding:utf-8
# 可视化检索系统
from PyQt5 import QtCore,QtGui,QtWidgets
import sys,os,threading,csv,time
import torch.nn as nn
import torch
sys.path.append("..")
from models import Corr_AE,Corr_CAE
from data_processing import img_feature_get,text_feature_get
path = './icon/'
data_path = ''
texts = []
img_files = []
n = 2#信号量，保证数据读取完毕后再进行模型的训练

class TextLoaderThread(threading.Thread):
    def __init__(self):
        super(TextLoaderThread,self).__init__()
    def run(self):
        global _text_data,text_data
        text_data = text_feature_get.get_text_feature(texts=_text_data[:1000])
        mi = text_data.min().numpy()
        ma = text_data.max().numpy()
        text_data = (text_data - mi) / (ma - mi)
        #文本数据0-1归一化
        global n
        n -= 1
#文本信息读取线程

class ImgLoaderThread(threading.Thread):
    def __init__(self):
        super(ImgLoaderThread,self).__init__()
    def run(self):
        global _img_data,img_data,y
        img_data = img_feature_get.get_img_feature(files=_img_data[:5000],mode=3)
        global n
        n -= 1
#图像信息读取线程


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.model = Corr_CAE.Corr_CAE(text_size=6000,img_size=250*250)
        self.model.load()
        #读取模型
        global path

        self.setFixedSize(1400,800)
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        #设置窗口主部件


        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(self.left_layout)
        #设置左侧部件

        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName('right_widget')
        #设置右侧部件
        self.main_layout.addWidget(self.left_widget,0,0,12,2)
        self.main_layout.addWidget(self.right_widget,0,2,12,10)
        self.setCentralWidget(self.main_widget)
        #设置窗口主部件

        self.close_button = QtWidgets.QPushButton("",self)
        self.close_button.setFixedSize(30,30)
        self.close_button.move(1340,20)
        self.close_button.setIcon(QtGui.QIcon(path+'close.png'))
        self.close_button.setStyleSheet("QPushButton{border:none;border-radius:5px;}"
                                        "QPushButton:hover{background:red;}")
        #设置关闭图标

        self.text = QtWidgets.QLabel('多模态影像报告检索系统', self)
        self.text.setStyleSheet("QLabel{border:none;color:black;}")
        self.text.setFont(QtGui.QFont("Roman times",10,QtGui.QFont.Bold))
        self.text.setFixedSize(200, 20)
        self.text.move(30, 30)
        #设置系统标题

        self.search_way = QtWidgets.QPushButton("检索方式")
        self.search_way.setObjectName('left_label')

        self.img2text = QtWidgets.QPushButton("图检文")
        self.img2text.setObjectName('left_button')
        self.img2text.setIcon(QtGui.QIcon(path+'search.png'))
        self.text2img = QtWidgets.QPushButton("文检图")
        self.text2img.setObjectName('left_button')
        self.text2img.setIcon(QtGui.QIcon(path+'search.png'))
        self.img_and_text = QtWidgets.QPushButton("联合检索")
        self.img_and_text.setObjectName('left_button')
        self.img_and_text.setIcon(QtGui.QIcon(path+'search.png'))
        self.left_layout.addWidget(self.search_way, 1, 0, 1, 3)
        self.left_layout.addWidget(self.img2text,2,0,1,3)
        self.left_layout.addWidget(self.text2img,3,0,1,3)
        self.left_layout.addWidget(self.img_and_text,4,0,1,3)
        #设置左侧功能按钮

        self.open_text_file = QtWidgets.QPushButton("请选择报告文件",self)
        self.open_text_file.setStyleSheet("QPushButton{border:2px solid black;border-top-left-radius:8px;border-top-right-radius:8px;border-bottom-left-radius:8px;border-bottom-right-radius:8px;}"
                                          "QPushButton:hover{border:4px solid green;}")
        self.open_text_file.setFixedSize(400,50)
        self.open_text_file.setIcon(QtGui.QIcon(path+'choose.png'))
        self.open_img_file = QtWidgets.QPushButton("请选择图像文件",self)
        self.open_img_file.setStyleSheet("QPushButton{border:2px solid black;border-top-left-radius:8px;border-top-right-radius:8px;border-bottom-left-radius:8px;border-bottom-right-radius:8px;}"
                                         "QPushButton:hover{border:4px solid green;}")
        self.open_img_file.setFixedSize(400,50)
        self.open_img_file.setIcon(QtGui.QIcon(path + 'choose.png'))
        self.search = QtWidgets.QPushButton("开始搜索",self)
        self.search.setStyleSheet("QPushButton{border:none;color:black;}"
                                  "QPushButton:hover{border-left:2px solid blue;font-weight:10;}")
        self.search.setGeometry(300,150,100,20)
        self.search.setIcon(QtGui.QIcon(path+'search.png'))

        #设置右侧功能组件

        self.left_widget.setStyleSheet("QPushButton{border:none;color:black;}"
                                       "QPushButton#left_label{border:none;border-bottom:1px solid black;font-size:18px;font-weight:700;font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}"
                                        "QPushButton#left_button:hover{border-left:4px solid blue;font-weight:700;}"
                                       "QWidget#left_widget{background:#9FFFEF;border-top:1px solid darkGray;border-bottom:1px solid darkGray;border-left:1px solid darkGray;border-top-left-radius:10px;border-bottom-left-radius:10px}")

        self.right_widget.setStyleSheet('''QWidget#right_widget{
                        color:#232C51;
                        background:white;
                        border-top:1px solid darkGray;
                        border-bottom:1px solid darkGray;
                        border-right:1px solid darkGray;
                        border-top-right-radius:10px;
                        border-bottom-right-radius:10px;}''')
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.main_layout.setSpacing(0)

        self.img_labels = [QtWidgets.QLabel(self) for _ in range(3)]
        for i in range(3):
            self.img_labels[i].setGeometry(400 + 210 * i, 300, 200, 250)
            self.img_labels[i].setScaledContents(True)
        self.text_labels = [QtWidgets.QLabel(self) for _ in range(3)]
        for i in range(3):
            self.text_labels[i].setGeometry(400 + 260 * i, 300, 250, 300)
            self.text_labels[i].setStyleSheet("border: 1px solid black")
            self.text_labels[i].setScaledContents(True)
            self.text_labels[i].setWordWrap(True)
        #设置QLabel以展示检索结果

        self.flush()

        self.close_button.clicked.connect(lambda :self.close_window())
        self.img2text.clicked.connect(lambda :self.text_search())
        self.text2img.clicked.connect(lambda :self.img_search())
        self.img_and_text.clicked.connect(lambda :self.img_text_search())
        self.open_text_file.clicked.connect(lambda: self.open_file(text=True, img=False))
        self.open_img_file.clicked.connect(lambda: self.open_file(img=True, text=False))
        self.search.clicked.connect(lambda :self.get_result())
        #设置信号槽

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        #无边框

        self.img_file = None
        self.text_file = None


    def close_window(self):
            sys.exit()
    #退出窗口

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        point = event.windowPos()
        if point.x() > 1200 or point.y() > 50:
            return
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()
    #鼠标点击窗口上方区域进行拖动

    def open_file(self,text=False,img=False):
        if text and not img:
            file_name,file_type = QtWidgets.QFileDialog.getOpenFileName(self,"选取文件",os.getcwd(),"All Files(*);;Text Files(*.txt)")
            self.text_file = file_name
            if file_name:
                self.open_text_file.setText("已选文件:"+self.text_file)
        if img and not text:
            file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),"All Files(*);;Image Files(*.png)")
            self.img_file = file_name
            if file_name:
                self.open_img_file.setText("已选文件:"+self.img_file)
    #打开文件

    def flush(self):
        self.open_img_file.close()
        self.open_text_file.close()
        self.search.close()
        for label in self.img_labels:
            label.close()
        for label in self.text_labels:
            label.close()

    def img_search(self):
        self.flush()
        self.search.show()
        self.open_text_file.move(300,50)
        self.open_text_file.show()
        self.get_img = True
        self.get_text = False
    #文搜图
    def text_search(self):
        self.flush()
        self.search.show()
        self.open_img_file.move(300,50)
        self.open_img_file.show()
        self.get_text = True
        self.get_img = False
    #图搜文
    def img_text_search(self):
        self.flush()
        self.search.show()
        self.open_img_file.move(300,50)
        self.open_text_file.move(800,50)
        self.open_img_file.show()
        self.open_text_file.show()
        self.get_img = True
        self.get_text = True
    #联合搜索
    def get_result(self):
        img_result = None
        text_result = None
        for label in self.img_labels:
            label.close()
        for label in self.text_labels:
            label.close()
        global img_data,text_data,_img_data,_text_data
        if self.get_img and self.text_file and not self.get_text:
            f = open(self.text_file,'r')
            text = f.read()
            result = self.model.search_top3(mode=2,search_data=img_data,text=text_feature_get.get_text_feature([text])[0])
            for i in range(3):
                self.img_labels[i].setPixmap(QtGui.QPixmap(_img_data[result[i]]))
                self.img_labels[i].show()
        elif self.get_text and self.img_file and not self.get_img:
            result = self.model.search_top3(mode=1,search_data=text_data,img=img_feature_get.get_img_feature([self.img_file],mode=3)[0])
            for i in range(3):
                self.text_labels[i].setText(_text_data[result[i]])
                self.text_labels[i].show()
        else :
            f = open(self.text_file, 'r')
            text = f.read()
            result = self.model.search_top3(mode=3,search_data=text_data,text=text_feature_get.get_text_feature([text])[0],img=img_feature_get.get_img_feature([self.img_file],mode=3)[0])
            for i in range(3):
                self.text_labels[i].setText(_text_data[result[i]])
                self.text_labels[i].show()

def show_main_ui():
    global path
    path = os.path.dirname(os.path.abspath(__file__))+'/icon/'
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()

    x = input("input path of data:")
    texts = list(csv.reader(open(x + '/cxr/report/indiana_reports.csv', encoding='utf-8')))[1:]
    global _text_data
    _text_data = [texts[i][6] for i in range(len(texts)) if texts[i][6] != ""]
    imgs = list(csv.reader(open(x + '/cxr/report/indiana_projections.csv', encoding='utf-8')))[1:]

    global _img_data
    _img_data = []
    for i in range(len(imgs)):
        filename = 'CXR' + imgs[i][1].replace('.dcm', '')
        _img_data.append(x + '/cxr/image/' + filename)
        _img_data.append(x + '/cxr/image/' + 'flip_' + filename)

    global text_data
    global img_data
    t1 = TextLoaderThread()
    t2 = ImgLoaderThread()
    t1.start()
    t2.start()
    # 通过两个线程同时对图像数据和文本数据
    while n:
        time.sleep(5)
    # 每5秒主线程检查数据是否读取完毕

    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__)) + '/icon/'
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())
