#Import thu vien
import wx
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
from test import data, y_pred, y_pred2, y, X, thongke, thongke2, thongke3, testpoint, testpoint2
import joblib


#Tao frame app
class MyFrame(wx.Frame):
    def __init__(self, parent, title): 
        super(MyFrame, self).__init__(parent, title = title, size=(700,400))

        # Create the menubar
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")
        self.Bind(wx.EVT_MENU, self.OnTimeToClose, id=wx.ID_EXIT)
        menuBar.Append(menu, "&File")
        self.SetMenuBar(menuBar)
        self.CreateStatusBar()
		
        panel = wx.Panel(self) 
        vbox = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(panel, -1, "GIAO DIEN USER")
        text.SetForegroundColour(wx.Colour(250,101,0))
        text.SetFont(wx.Font(20, wx.SWISS, wx.NORMAL, wx.BOLD))
        text.SetSize(text.GetBestSize())
        tkbtn = wx.Button(panel, -1, "Thong ke")
        btn = wx.Button(panel, -1, "Close")
        boxall=wx.BoxSizer(wx.HORIZONTAL)


        #Box button train model no PCA
        nm = wx.StaticBox(panel, -1, 'Train model no PCA:') 
        nmSizer = wx.StaticBoxSizer(nm, wx.VERTICAL) 
        nmbox = wx.BoxSizer(wx.VERTICAL)
      
        chbtn = wx.Button(panel, -1, "Thong ke sau chuan hoa")
        checkbtn = wx.Button(panel, -1, "Accuracy-Report and Confuse-Matrix")
        testbtn = wx.Button(panel, -1, "Test chon mau")
        nmbox.Add(chbtn, 0, wx.ALL|wx.CENTER, 10)  
        nmbox.Add(checkbtn, 0, wx.ALL|wx.CENTER, 10)  
        nmbox.Add(testbtn, 0, wx.ALL|wx.CENTER, 10) 
        nmSizer.Add(nmbox, 0, wx.ALL|wx.CENTER, 10)  


        #Box button train model with PCA		
        sbox = wx.StaticBox(panel, -1, 'Train model with PCA:') 
        sboxSizer = wx.StaticBoxSizer(sbox, wx.VERTICAL) 
        hbox = wx.BoxSizer(wx.VERTICAL) 
      
        ch2btn = wx.Button(panel, -1, "Thong ke sau chuan hoa")
        check2btn = wx.Button(panel, -1, "Accuracy-Report and Confuse-Matrix")
        test2btn = wx.Button(panel, -1, "Test chon mau")
        hbox.Add(ch2btn, 0, wx.ALL|wx.CENTER, 10) 
        hbox.Add(check2btn, 0, wx.ALL|wx.CENTER, 10)
        hbox.Add(test2btn, 0, wx.ALL|wx.CENTER, 10)  
        sboxSizer.Add(hbox, 0, wx.ALL|wx.CENTER, 10)


        #Build chuc nang cho button
        self.Bind(wx.EVT_BUTTON, self.OnTimeToClose, btn)
        self.Bind(wx.EVT_BUTTON, self.OnAccuracyButton, checkbtn)
        self.Bind(wx.EVT_BUTTON, self.OnAccuracy2Button, check2btn)
        self.Bind(wx.EVT_BUTTON, self.OnTestButton, testbtn)
        self.Bind(wx.EVT_BUTTON, self.OnTest2Button, test2btn)
        self.Bind(wx.EVT_BUTTON, self.OnTKButton, tkbtn)
        self.Bind(wx.EVT_BUTTON, self.OnCHTKButton, chbtn)
        self.Bind(wx.EVT_BUTTON, self.OnCHTK2Button, ch2btn)

        
        #Add box train va title vo panel
        boxall.Add(nmSizer, 0, wx.ALL|wx.CENTER, 5) 
        boxall.Add(sboxSizer, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(text, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(tkbtn, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(boxall, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(btn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(vbox)
        panel.SetBackgroundColour(wx.Colour(115,175,225)) 
        self.Centre() 
         
        panel.Fit() 
        self.Show()



    #Button dong chuong trinh
    def OnTimeToClose(self, evt):
        print ("End!")

        self.Close()

    #Button tinh do chinh xac va tao confuse-martix
    def OnAccuracyButton(self, evt):
        testpoint()
        print("\n")
    

    #Button tinh do chinh xac va tao confuse-martix PCA
    def OnAccuracy2Button(self, evt):
        testpoint2()
        print("\n")
    
    #Button thuc hien thong ke so bo dataset
    def OnTKButton(self, evt):
        thongke()
        print("\n")
    

    #Button thuc hien thong ke sau khi chuan hoa
    def OnCHTKButton(self, evt):
        thongke2()
        print("\n")
    

    #Button thuc hien thong ke sau khi chuan hoa PCA
    def OnCHTK2Button(self, evt):
        thongke3()
        print("\n")


    #Button thuc hien tao cua so nhap va test mau
    def OnTestButton(self, evt):
        text=wx.TextEntryDialog(self, "Nhap so thu tu mau can test(STT trong csv-2 VD: (STT 5 nhap 3)):", "Window Nhap")
        if text.ShowModal()==wx.ID_OK:
            num=int(text.GetValue())
            print("Mau so: ", num," dua tren file .csv")
            print("True label: ", data.y.iloc[num-2])
            if(y_pred[num-2]==1):
                print("Label with train model: Yes")
            else:
                print("Label with train model: No")
    
    def OnTest2Button(self, evt):
        text=wx.TextEntryDialog(self, "Nhap so thu tu mau can test(STT trong csv-2 VD: (STT 5 nhap 3)):", "Window Nhap")
        if text.ShowModal()==wx.ID_OK:
            num=int(text.GetValue())
            print("Mau so: ", num , " dua tren file .csv")
            print("True label: ", data.y.iloc[num-2])
            if(y_pred2[num-2]==1):
                print("Label with train model: Yes")
            else:
                print("Label with train model: No")
    

        

#Control app
class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, "Bank Marketing App")
        self.SetTopWindow(frame)

        print("WINDOW OUTPUT")

        frame.Show(True)
        return True
        
app = MyApp(redirect=True)
app.MainLoop()