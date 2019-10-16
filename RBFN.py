# -*- coding: utf-8 -*-
import numpy as np;
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import math
from PyQt5 import QtWidgets, QtCore
from mpl_toolkits.mplot3d import Axes3D

#Canvas
class Figure_Canvas(FigureCanvas):

    def __init__(self, parent=None, width=4.7, height=4.7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=100)  

        FigureCanvas.__init__(self, self.fig) 
        self.setParent(parent)

        self.axes = self.fig.add_subplot(111) 

    def test(self, data, w, n, m, d, dim):
        #Plot point
        color = {
                    0:'red',
                    1:'blue',
                    2:'yellow',
                    3:'green',
                    4:'purple',
                    5:'black'
                }
        if dim == 2: 
            for i in range(n):
                self.axes.scatter(data[i, 0], data[i, 1], color=color[int(data[i, 2]%5)], s = 15, alpha=0.8)
            '''
            #Plot line
            a = np.arange(-10, 10, 0.1)
            b =( w[0] + -1*w[1]*a ) / w[2] 
            self.axes.plot(a, b)
            '''
            for i in range(len(m)) :
                circle1 = plt.Circle((m[i][0], m[i][1]), abs(d[i]), color='g', fill=False)
                self.axes.add_artist(circle1)
            size_max = max(max(data[:, 0]), max( data[:, 1])) + 1
            size_min = min(min(data[:, 0]), min( data[:, 1])) - 1
            self.axes.set_xlim(size_max, size_min)
            self.axes.set_ylim(size_max, size_min)
        elif dim == 3:
            self.axes = Axes3D(self.fig)
            for i in range(n):
                self.axes.scatter(data[i, 0], data[i, 1], data[i, 2], color=color[int(data[i, 3])], s = 15, alpha=0.8)
            for i in range(len(m)) :
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x =  abs(d[i]) * np.outer(np.cos(u), np.sin(v)) + m[i][0]
                y =  abs(d[i])* np.outer(np.sin(u), np.sin(v)) + m[i][1]
                z =  abs(d[i]) * np.outer(np.ones(np.size(u)), np.cos(v)) + m[i][2]
                self.axes.plot_wireframe(x, y, z, color="black", alpha = 0.02)

 
#Training
class Train():
    
    def __init__(self):
        self.fileName = ""
        self.rate = 0.0
        self.gp = 0
        self.progress_bar = QtWidgets.QProgressBar()
        self.train_acc_text = ""
        self.test_acc_text = ""
        self.weight = ""
        self.rmse_text = ""
    
    def set(self, fileName, rate, rnd, group, gui):
        self.fileName = fileName
        self.rate = rate
        self.round = rnd
        self.gn = group
        self.gui = gui

    def run(self):
        group_num = self.gn
        #轉換label
        label = []
        #Read data
        fi=open(self.fileName, 'r')
        #Set iteration
        it = self.round
        #Set learning rate
        lr = self.rate
        lr_o = lr
        #計算維度
        line=fi.readline()
        line = line.split()
        dim = len(line) - 1       #資料維度
        #讀取數據用
        data = np.empty(shape=[0, dim+1])
        #把第一筆加上去
        #line = np.hstack((['-1'], line))
        #更改label
        label.append(line[-1])
        line[-1] = label.index(line[-1])
        data = np.row_stack((data, line))
        #資料筆數
        n = 1
        while 1:
            line=fi.readline()
            if line=="":
                break
            line = line.split()
            #line = np.hstack((['-1'], line))
            #更改label
            if label.count(line[-1]) == 0 :
                label.append(line[-1])
            line[-1] = label.index(line[-1])
            #########
            data = np.row_stack((data, line))
            n = n + 1
        data = data.astype(np.float)
        cls = len(label)
        
        #切割資料
        train_data = np.empty(shape=[0, dim+1])
        test_data = np.empty(shape=[0, dim+1])
        test_num = random.sample(range(len(data)), k = int(len(data) / 3) )
        for i in range(len(data)):
            if i in test_num:
                test_data = np.row_stack((test_data, data[i]))
            else:
                train_data = np.row_stack((train_data, data[i]))
        
        #Find group central
        #Initailize central
        ctn = 1
        count = 0
        while(ctn == 1 and count < 50):
            group_central = []
            for i in range(group_num) :
                tmp = []
                for k in range(dim) :
                    tmp.append(random.uniform(max(data[:, 1]), min(data[:, 1])))
                group_central.append(tmp)
        
            group_ori=[ [] for i in range(group_num) ]
            for i in range(100) :
                group=[ [] for i in range(group_num) ]
                for k in range(len(train_data)) :
                    min_dist = sys.maxsize
                    for m in range(group_num) :
                        tmp = dist(train_data[k, 0:dim], group_central[m], dim)
                        if tmp < min_dist :
                            min_dist = tmp
                            tmp_group = m     
                    group[tmp_group].append(k)
                #Calculate new central
                for m in range(group_num) :
                    sum = [0]*(dim)
                    for k in group[m]:
                       sum = [sum[n]+train_data[k][n] for n in range(dim)]
                    if len(group[m]) != 0 :
                        group_central[m] = [sum[n]/(len(group[m])) for n in range(len(sum))]
                #if same then break
                if group_ori == group:
                    break
                else:
                    group_ori = group
                ctn = 0
                for i in range(group_num) :
                    if len(group[i]) == 0:
                        ctn = 1
                        count += 1
                        break
        #for i in range(group_num):
            #print(group[i])
        
        #Set Weight 
        w = np.array([random.uniform(0, 1)] * (group_num+1))
        
        
        
        #set delta
        delta = [0]*group_num
        for i in range(group_num) :    
            for k in group[i]:
                           delta[i] += math.pow(dist(train_data[k, 0:dim], group_central[i], dim), 2)
            if len(group[i]) != 0 :               
                delta[i] = delta[i] / len(group[i])
                delta[i] = math.sqrt(delta[i])
            if delta[i] == 0:
                delta[i] = 0.1
        
        #Training
        b_acc = 0
        for i in range(it) :
            for k in range(len(train_data)) :
                phi = np.array([1])
                for m in range(group_num) :
                    phi_m = 0
                    if delta[m]!= 0:
                        phi_m = math.exp((-1)*math.pow(dist(train_data[k, 0:dim], group_central[m], dim), 2) / (2*math.pow(delta[m], 2)))
                    phi = np.hstack((phi, [phi_m]))
                f = np.dot(phi, w)
                #把舊的記下來因為等等會更動
                w_o = w         
                group_central_o = group_central 
                #開始更動
                w = w + lr*(train_data[k, dim] - f)/(dim-1)*phi
                for n in range(group_num) :
                    for m in range(len(group_central[n])) :
                        group_central[n][m] = group_central[n][m] + lr*(train_data[k, dim] - f)/(dim-1)*w_o[n+1]*phi[n+1]/math.pow(delta[n], 2)*(train_data[k][m]-group_central[n][m])
                for m in range(len(delta)) :
                    delta = delta + lr*(train_data[k, dim] - f)/(dim-1)*w_o[m+1]*phi[m+1]/math.pow(delta[m], 3)*math.pow(dist(train_data[k, 0:dim], group_central_o[m], dim), 2)
            lr = lr_o * ( ( it - i ) / it)
            #計算測試精準度
            cnt = 0
            for k in range(len(train_data)):
                phi = np.array([1])
                for m in range(group_num) :
                    phi_m = math.exp((-1)*math.pow(dist(train_data[k, 0:dim], group_central[m], dim), 2) / (2*math.pow(delta[m], 2)))
                    phi = np.hstack((phi, [phi_m]))
                f = sgn(np.dot(phi, w), cls)
                if sgn(f, cls) == int(train_data[k, dim]):
                    cnt = cnt + 1
            train_acc = cnt/len(train_data)
            if train_acc >= b_acc :
                b_acc = train_acc
                b_delta = delta
                b_w = w
                b_group_central = group_central
            if b_acc == 1 :
                break
        #計算測試精準度
        cnt = 0
        rmse = 0
        for k in range(len(test_data)):
            phi = np.array([1])
            for m in range(group_num) :
                phi_m = math.exp((-1)*math.pow(dist(test_data[k, 0:dim], b_group_central[m], dim), 2) / (2*math.pow(b_delta[m], 2)))
                phi = np.hstack((phi, [phi_m]))
            f = sgn(np.dot(phi, b_w), cls)
            rmse += math.pow(f - test_data[k, dim],2)
            if sgn(f, cls) == int(test_data[k, dim]):
                cnt = cnt + 1
        rmse = rmse/len(test_data)
        test_acc = cnt/len(test_data)
        
        #顯示權重
        self.weight = str(b_w)
        #顯示精準度
        self.train_acc_text = str(b_acc * 100)
        self.test_acc_text = str(test_acc * 100)
        #print(test_acc)
        #print(delta)
        #顯示RMSE
        self.rmse_text = str(rmse)
        
        '''
        #Plot point 
        color = {
                    0:'red',
                    1:'blue',
                    2:'yellow',
                    3:'green' }
        for i in range(len(train_data)):
            plt.scatter(train_data[i, 0], train_data[i, 1], color=color[train_data[i, dim]], s = 15, alpha=0.8)
        for i in range(group_num) :
            plt.scatter(group_central[i][0], group_central[i][1], color = 'green', s = 90, alpha=0.8)
        plt.show()
        '''
        self.train_dr = Figure_Canvas()
        self.test_dr = Figure_Canvas()

        self.train_dr.test(train_data, b_w, len(train_data), b_group_central, b_delta, dim)
        self.test_dr.test(test_data, b_w, len(test_data), b_group_central, b_delta, dim)

          
        fi.close()
        
    def get_test_pic(self):
        return self.test_dr
    
    def get_train_pic(self):
        return self.train_dr

#GUI
class Input(QtWidgets.QWidget):

    def __init__(self, parent = None):

        super().__init__(parent)
        self.progress_bar = []
        self.fileName = ""
        self.rate = 0.0
        self.round = 0
        self.count = 0
        self.group = 0

        self.layout = QtWidgets.QFormLayout()
        self.Label1 = QtWidgets.QLabel("File name")
        self.tmp1 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label1, self.tmp1)

        self.Label2 = QtWidgets.QLabel("Learning rate")
        self.tmp2 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label2, self.tmp2)
        
        self.Label3 = QtWidgets.QLabel("Round")
        self.tmp3 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label3, self.tmp3)
        
        self.Label9 = QtWidgets.QLabel("Group num")
        self.tmp4 = QtWidgets.QLineEdit()
        self.layout.addRow(self.Label9, self.tmp4)

        self.btn = QtWidgets.QPushButton('Ok')
        self.btn.clicked.connect(self.grab)
        self.layout.addRow(self.btn)
        
        self.Label4 = QtWidgets.QLabel("Train")
        self.Label5 = QtWidgets.QLabel("Test")
        self.layout.addRow(self.Label4, self.Label5)
        
        self.graphicview = QtWidgets.QGraphicsView()
        self.graphicview2 = QtWidgets.QGraphicsView()
        self.graphicview.setFixedSize(500, 500)
        self.layout.addRow(self.graphicview, self.graphicview2)
        
        
        self.Label7 = QtWidgets.QLabel("Test accuracy: ")
        self.Label6 = QtWidgets.QLabel("Train accuracy: ")
        self.layout.addRow(self.Label6, self.Label7)
        
        self.Label8 = QtWidgets.QLabel("Weight: ")
        self.Label10 = QtWidgets.QLabel("RMSE: ")
        self.layout.addRow(self.Label8, self.Label10)
        
        self.setLayout(self.layout)
        self.setWindowTitle("HW2")
        self.setGeometry(100, 150, 1050, 800)
        
    def grab(self):             #多線程處理
        print("Get process!")
        self.fileName = self.tmp1.text()
        self.rate = float(self.tmp2.text())
        self.round = int(self.tmp3.text())
        self.group = int(self.tmp4.text())
        self.arrange = False
        self.check()
        
    def check(self):
        if self.rate <= 0 or self.round <= 0 or self.group <= 0:
            print("Invalid input")
            return None

        self.train = Train()
        self.train.set(self.fileName, self.rate, self.round, self.group, self)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene2 = QtWidgets.QGraphicsScene()
        self.train.run()
        graphicscene.addWidget(self.train.get_train_pic())
        graphicscene2.addWidget(self.train.get_test_pic())
        self.graphicview.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicview2.setScene(graphicscene2)
        self.graphicview.show()
        self.graphicview2.show()
        self.Label6.setText("Train accuracy: " + self.train.train_acc_text + "%")
        self.Label7.setText("Test accuracy: " + self.train.test_acc_text + "%")
        self.Label8.setText("Weight: " + self.train.weight)
        self.Label10.setText("RMSE: " + self.train.rmse_text)

def main():
    
    app = QtWidgets.QApplication(sys.argv)
    window = Input()
    window.show()
     
    #close window 
    app.exec_()

def dist(x, y, z):  #x,y:點 z:維度
    d = 0
    for i in range(z) :
        d += math.pow(x[i] - y[i], 2) 
    d = math.sqrt(d)
    return d

def sgn(x, cls):
    x = int(x / ((cls-1)/cls))
    while x>=cls :
        x = x-1
    return x
    
if __name__ == "__main__":  
    main()
