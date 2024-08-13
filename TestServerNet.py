import pygame
import os
import sys
import random
import time
import gc
import numpy as np
from torch import multiprocessing as mp
import torch
from torch.autograd import Variable
import copy
import socket
import logging
from pygame.locals import *
from time import sleep

from try_resnet_0706 import BasicBlock
from try_resnet_0713_value import ResNet as ResNetValue

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
torch.backends.cudnn.benchmark = True

WINDOWSIZE = (1400, 680)  # 游戏窗口大小
LINECOLOR = (0, 0, 0)  # 棋盘线的颜色
TEXTCOLOR = (0, 0, 0)  # 标题文本颜色
BLACKGROUND = (255, 255, 255)  # 游戏背景颜色
BLACK = (0, 0, 0)
BLUE = (0, 191, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
SIZE = (50, 50)  # 棋子大小
AYSTSIZE = (240, 240)  # 爱因斯坦背景头像图片大小
POSAYST = (365, 0)  # 爱因斯坦头像位置
STEP = 60  # 步长：棋子走一步移动的距离
TEXTSIZE = 30  # 标题文字大小
TIPSIZE = 15  # 提示文字大小
COUNT = -1  # 记录当前回合
START = 65  # 棋盘左上角起始点（70，70）
REDWIN = 1  # 代表RED方赢
BLUEWIN = 2  # 代表玩家BLUE方赢
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
LEFTUP = 'leftup'
RIGHTDOWN = 'rightdown'
RESULT = [0, 0]  # 记录比赛结果
WINSIZE = (530, 130)  # 显示比赛结果窗口大小
INFTY = 10000
SLEEPTIME = 0
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s- %(levelname)s - %(message)s', filename='TestServer0808.log')
logger = logging.getLogger()
ch = logging.StreamHandler() #日志输出到屏幕控制台
ch.setLevel(logging.INFO) #设置日志等级
formatter = logging.Formatter('%(asctime)s %(name)s- %(levelname)s - %(message)s') #定义日志输出格式
ch.setFormatter(formatter) #选择一个格式
logger.addHandler(ch) #增加指定的handler

class Status(object):
    def __init__(self):
        self.map = None
        self.value = None
        self.pawn = None
        self.pro = None
        self.parent = None
        self.parent_before = None
        self.parent_3 = None
        self.parent_4 = None
        self.pPawn = None
        self.pMove = None
        self.pDice = None
        self.cPawn = None
        self.cPawnSecond = None
        self.cPM = [[],[],[],[],[],[]]
        self.cPMSecond = [[],[],[],[],[],[]]
    def print(self):
        print(self.cPM)
    def __str__(self):
        # print(Status)
        return '[棋子为%s , 选择方向为%s]' % (self.pPawn, self.pMove)

def init():
    global IMAGE,tip,screen,font,maplib,Lyr,Lyb,Lx,S,matchPro
    pygame.init()
    S = Status()
    screen = pygame.display.set_mode(WINDOWSIZE, 0, 32)  # 设置游戏窗口
    pygame.display.set_caption('EinStein wurfelt nicht')  # 设置Caption
    font = pygame.font.SysFont("Cambria Math", TEXTSIZE,
                               bold=False, italic=False)  # 设置标题字体格式
    tip = pygame.font.SysFont("arial", TIPSIZE, bold=False, italic=False)  # 设置提示字体
    IMAGE = {
        'R1': pygame.transform.scale(pygame.image.load('picture/white/R1.png').convert(), SIZE),
        'R2': pygame.transform.scale(pygame.image.load('picture/white/R2.png').convert(), SIZE),
        'R3': pygame.transform.scale(pygame.image.load('picture/white/R3.png').convert(), SIZE),
        'R4': pygame.transform.scale(pygame.image.load('picture/white/R4.png').convert(), SIZE),
        'R5': pygame.transform.scale(pygame.image.load('picture/white/R5.png').convert(), SIZE),
        'R6': pygame.transform.scale(pygame.image.load('picture/white/R6.png').convert(), SIZE),
        'B1': pygame.transform.scale(pygame.image.load('picture/white/B1.png').convert(), SIZE),
        'B2': pygame.transform.scale(pygame.image.load('picture/white/B2.png').convert(), SIZE),
        'B3': pygame.transform.scale(pygame.image.load('picture/white/B3.png').convert(), SIZE),
        'B4': pygame.transform.scale(pygame.image.load('picture/white/B4.png').convert(), SIZE),
        'B5': pygame.transform.scale(pygame.image.load('picture/white/B5.png').convert(), SIZE),
        'B6': pygame.transform.scale(pygame.image.load('picture/white/B6.png').convert(), SIZE),
        'Y1': pygame.transform.scale(pygame.image.load('picture/white/Y1.png').convert(), SIZE),
        'Y2': pygame.transform.scale(pygame.image.load('picture/white/Y2.png').convert(), SIZE),
        'Y3': pygame.transform.scale(pygame.image.load('picture/white/Y3.png').convert(), SIZE),
        'Y4': pygame.transform.scale(pygame.image.load('picture/white/Y4.png').convert(), SIZE),
        'Y5': pygame.transform.scale(pygame.image.load('picture/white/Y5.png').convert(), SIZE),
        'Y6': pygame.transform.scale(pygame.image.load('picture/white/Y6.png').convert(), SIZE),
        '1': pygame.transform.scale(pygame.image.load('picture/white/1.png').convert(), SIZE),
        '2': pygame.transform.scale(pygame.image.load('picture/white/2.png').convert(), SIZE),
        '3': pygame.transform.scale(pygame.image.load('picture/white/3.png').convert(), SIZE),
        '4': pygame.transform.scale(pygame.image.load('picture/white/4.png').convert(), SIZE),
        '5': pygame.transform.scale(pygame.image.load('picture/white/5.png').convert(), SIZE),
        '6': pygame.transform.scale(pygame.image.load('picture/white/6.png').convert(), SIZE),
        'BLUEWIN': pygame.transform.scale(pygame.image.load('picture/white/BLUEWIN.png').convert(), WINSIZE),
        'REDWIN': pygame.transform.scale(pygame.image.load('picture/white/REDWIN.png').convert(), WINSIZE),
    }
    # 布局库
    maplib = [
        [6, 1, 3, 2, 5, 4],
        [6, 2, 4, 1, 5, 3],
              [6, 5, 2, 1, 4, 3],
              [1, 5, 4, 6, 2, 3],
              [1, 6, 3, 5, 2, 4],
              [1, 6, 4, 3, 2, 5],
              [6, 1, 2, 5, 4, 3],
              [6, 1, 3, 5, 4, 2],
              [1, 6, 4, 2, 3, 5],
              [1, 5, 2, 6, 3, 4],
              [1, 6, 5, 2, 3, 4],
              [1, 2, 5, 6, 3, 4],
              [6, 2, 5, 1, 4, 3],
              [1, 6, 3, 2, 4, 5],
              [6, 2, 3, 1, 5, 4],
              [1, 6, 3, 4, 2, 5],
              [1, 5, 4, 6, 3, 2]
              ]
    # resetInfo()
    Lyr = []
    Lyb = []
    Lx = []
    matchPro = 0.85

def loadImage(name, pos, size=SIZE):
    filename = "picture/white/" + name
    screen.blit(pygame.transform.scale(
        pygame.image.load(filename).convert(), size), pos)

def waitForPlayerToPressKey():  # 等待按键
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    terminate()
                return

def drawStartScreen():  # 开始界面
    screen.fill(WHITE)
    loadImage("AYST.png", (190, 40), AYSTSIZE)
    drawText('127.0.0.1:50007 IS WAITING FOR CONNECTION', font, TEXTCOLOR, screen, 6.5, 2)
    pygame.display.update()


    # waitForPlayerToPressKey()

def drawWinScreen(result):  # 比赛结束，显示结果界面
    if result == BLUEWIN:
        loadImage("BLUEWIN.png", (50, 500), WINSIZE)
    if result == REDWIN:
        loadImage("REDWIN.png", (50, 500), WINSIZE)
    # waitForPlayerToPressKey()
    pygame.display.update()
    sleep(SLEEPTIME)

def showWinRate(RedWinRate, BlueWinRate, x):
    global Lyr, Lyb, Lx
    yr = (100 - RedWinRate)/(100/3.0) + 0.6
    yb = (100 - BlueWinRate)/(100/3.0) + 0.6
    x = x/(1000/5) + 4.2
    Lyr.append(copy.deepcopy(yr))
    Lyb.append(copy.deepcopy(yb))
    Lx.append(copy.deepcopy(x))
    for i in range(0, len(Lyr)-1):
        pygame.draw.line(
            screen, RED, (100*Lx[i], 100*Lyr[i]), (100*Lx[i], 100*Lyr[i+1]))
        pygame.draw.line(
            screen, BLUE, (100*Lx[i], 100*Lyb[i]), (100*Lx[i], 100*Lyb[i+1]))

def drawGameScreen(Red, Blue):  # 游戏比赛界面
    global S
    screen.fill(WHITE)
    # 画棋盘
    for i in range(6):
        x = y = 60*(i+1)
        pygame.draw.line(screen, LINECOLOR, (60, y), (360, y))
        pygame.draw.line(screen, LINECOLOR, (x, 60), (x, 360))
    # 加载提示文字
    drawText('Winning Percentage Dynamic Figure', font, BLACK, screen, 0, 7.2)
    drawText(Red + ' Vs ' + Blue, font, BLACK, screen, 0.5, 7.2)
    drawText('matchPro : '+ str(round(100*matchPro,4))+'%',font,BLACK,screen,1,7.2)


    # 胜率坐标轴
    pygame.draw.line(screen, LINECOLOR, (415, 55), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (425, 55), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (420, 360), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (420, 360), (1200, 360))
    pygame.draw.line(screen, LINECOLOR, (1195, 355), (1200, 360))
    pygame.draw.line(screen, LINECOLOR, (1195, 365), (1200, 360))


    drawText('You can move: ', font, BLACK, screen, 0.1, 0.2)
    drawText('A : down', tip, BLACK, screen, 6, 1)
    drawText('W : right', tip, BLACK, screen, 6, 2.5)
    drawText('S : right-down', tip, BLACK, screen, 6, 4)
    drawText('U : up', tip, BLACK, screen, 6.5, 1)
    drawText('H : left', tip, BLACK, screen, 6.5, 2.5)
    drawText('Y : left-up', tip, BLACK, screen, 6.5, 4)
    drawText('RED : '+str(RESULT[0]), font, RED, screen, 6, 7)
    drawText('BLUE : '+str(RESULT[1]), font, BLUE, screen, 6.5, 7)
    drawText('RED : ' + Red, font,RED,  screen, 7.5, 2)
    drawText('BLUE: ' + ID, font,BLUE,  screen, 8, 2)
    if(sum(RESULT)):
        RedWinRate = round(100*float(RESULT[0])/sum(RESULT), 2)
        BlueWinRate = round(100*float(RESULT[1])/sum(RESULT), 2)
        drawText('RedWinRate:'+str(round(100 *
                                         float(RESULT[0])/sum(RESULT), 2)), font, RED, screen, 6, 9.5)
        drawText('BlueWinRate:'+str(round(100 *
                                          float(RESULT[1])/sum(RESULT), 2)), font, BLUE, screen, 6.5, 9.5)
        x = sum(RESULT)
        showWinRate(RedWinRate, BlueWinRate, x)
    # 画棋子
    for i in range(5):
        for j in range(5):
            if S.map[i][j] != 0:
                drawPawn(S.map[i][j], i, j)

    pygame.display.update()

def drawMovePawn(n, ans):  # 可选择移动的棋子
    x = -1
    y = 2
    for v in ans:
        drawPawn(v, x, y)
        y += 1
    if n <= 6:
        loadImage(str(n)+'.png', (310, 5))
    else:
        loadImage(str(n-6)+'.png', (310, 5))
    pygame.display.update()

def drawPawn(value, row, col, size=SIZE):  # 在（row，col）处，画值为value的棋子
    pos_x = col * STEP + START
    pos_y = row * STEP + START
    Pos = (pos_x, pos_y)
    if value <= 6:
        s = 'R' + str(value)
    elif value > 6:
        s = 'B' + str(value-6)
    loadImage(s+'.png', Pos, size)

def drawText(text, font, color, surface, row, col):  # 处理需要描绘的文字：text：文本；font：格式；
    row += 0.2
    x = col * STEP
    y = row * STEP
    textobj = font.render(text, True, color, WHITE)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def selectPawn(S,n=0):  # 掷骰子，挑选可以移动的棋子
    global COUNT
    if n == 0:
        COUNT += 1
        if COUNT % 2 == 0:
            n = random.randint(1, 6)
        else:
            n = random.randint(7, 12)
        ans = findNearby(n, S.pawn)
    else:
        ans = findNearby(n, S.pawn)
    return n, ans

def terminate():  # 退出游戏
    pygame.quit()
    sys.exit()

def makeMove(p, PawnMoveTo):  # 移动棋子，更新地图信息，和棋子存活情况
    back_S = copy.deepcopy(S)
    row, col = getLocation(p, S.map)
    x = y = 0
    if PawnMoveTo == LEFT:
        y = -1
    elif PawnMoveTo == RIGHT:
        y = +1
    elif PawnMoveTo == UP:
        x = -1
    elif PawnMoveTo == DOWN:
        x = +1
    elif PawnMoveTo == LEFTUP:
        x = -1
        y = -1
    elif PawnMoveTo == RIGHTDOWN:
        x = +1
        y = +1
    else:
        return False
    # 移动无效
    if notInMap(row+x, col+y):
        return False
    S.map[row][col] = 0
    row = row + x
    col = col + y
    # 是否吃掉自己或对方的棋
    if S.map[row][col] != 0:
        i = S.pawn.index(S.map[row][col])
        S.pawn[i] = 0
    S.map[row][col] = p
    value = getLocValue(S)  # 获取所有棋子的位置价值
    S.pro = getPawnPro(S)  # 获取所有棋子被摇到的概率
    S.value = getPawnValue(value, S.pro)
    S.parent = back_S
    if back_S.parent is not None:
        S.parent_before = back_S.parent.map
    else:
        S.parent_before = None
    S.parent_3 = back_S.parent_before
    S.parent_4 = back_S.parent_3
    return True

def notInMap(x, y):  # 检测棋子是否在棋盘内移动
    if x in range(0, 5) and y in range(0, 5):
        return False
    return True

def isEnd(S):  # 检测比赛是否结束
    if S.map[0][0] > 6:
        return BLUEWIN
    elif S.map[4][4] > 0 and S.map[4][4] <= 6:
        return REDWIN
    cnt = 0
    for i in range(0, 6):
        if S.pawn[i] == 0:
            cnt += 1
    if cnt == 6:
        return BLUEWIN
    cnt = 0
    for i in range(6, 12):
        if S.pawn[i] == 0:
            cnt += 1
    if cnt == 6:
        return REDWIN
    return False

def resetInfo():  # 重置比赛信息
    S.map = getNewMap()
    S.pawn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 棋子初始化
    S.pro = [1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0 /
           6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6]
    # value = getLocValue(S)
    # S.value = getPawnValue(S.pro, value)

def getNewMap():  # 换新图
    # r = random.sample(maplib, 1)[0]
    b = maplib[0]
    newMap = [
        [6, 2, 4, 0, 0],
        [1, 5, 0, 0, 0],
        [3, 0, 0, 0, b[2] + 6],
        [0, 0, 0, b[4] + 6, b[1] + 6],
        [0, 0, b[5] + 6, b[3] + 6, b[0] + 6]
    ]
    return newMap

def getPawnPro(S):  # 返回棋子被摇到的概率
    value = getLocValue(S)
    pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(1, 13):
        pro[p - 1] = 1.0 / 6
    for p in range(1, 13):
        if S.pawn[p - 1] == 0:
            ans = findNearby(p, S.pawn)
            if len(ans) > 1:
                pr = ans[0] - 1
                pl = ans[1] - 1
                if value[pr] > value[pl]:
                    pro[pr] += pro[p - 1]
                elif value[pr] == value[pl]:
                    pro[pr] += pro[p - 1] / 2
                    pro[pl] += pro[p - 1] / 2
                else:
                    pro[pl] += pro[p - 1]
            elif len(ans) == 1:
                pro[ans[0] - 1] += pro[p - 1]
            elif len(ans) == 0:
                pass
            pro[p - 1] = 0
    return pro

def getLocValue(S):  # 棋子所在位置的价值
    blueValue = [[99, 10,  6,  3,  1],
                 [10,  8,  4,  2,  1],
                 [6,  4,  4,  2,  1],
                 [3,  2,  2,  2,  1],
                 [1,  1,  1,  1,  1]]
    redValue = [[1,  1,  1,  1,  1],
                [1,  2,  2,  2,  3],
                [1,  2,  4,  4,  6],
                [1,  2,  4,  8, 10],
                [1,  3,  6, 10, 99]]
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(1, 13):
        if S.pawn[p-1] != 0:
            row, col = getLocation(p, S.map)
            if p <= 6:
                V[p-1] = redValue[row][col]
            else:
                V[p-1] = blueValue[row][col]
    return V

def getPawnValue(pro, value):  # 棋子价值
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 12):
        V[i] = pro[i] * value[i]
    return V

def searchNearbyBlueMaxValue(p, S):  # 搜索附近蓝方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row+1 < 5:
        if S.map[row+1][col] > 6:
            nearby.append(S.value[S.map[row+1][col]-1])
    if col+1 < 5:
        if S.map[row][col+1] > 6:
            nearby.append(S.value[S.map[row][col+1]-1])
    if row+1 < 5 and col+1 < 5:
        if S.map[row+1][col+1] > 6:
            nearby.append(S.value[S.map[row+1][col+1]-1])
    if nearby == []:
        return 0

    expValue = 0
    for v in nearby:
        expValue += v/sum(nearby)
    return expValue

def searchNearbyRedMaxValue(p, S):   # 搜索附近红方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row-1 >= 0:
        if S.map[row-1][col] <= 6 and S.map[row-1][col] > 0:
            nearby.append(S.value[S.map[row-1][col]-1])
    if col-1 >= 0:
        if S.map[row][col-1] <= 6 and S.map[row][col-1] > 0:
            nearby.append(S.value[S.map[row][col-1]-1])
    if row-1 >= 0 and col-1 >= 0:
        if S.map[row-1][col-1] <= 6 and S.map[row-1][col-1] > 0:
            nearby.append(S.value[S.map[row-1][col-1]-1])
    if nearby == []:
        return 0
    expValue = 0
    for v in nearby:
        expValue += v/sum(nearby)
    return expValue

def getThread(S):  # 获得红方对蓝方的威胁值，蓝方对红方的威胁值
    redToBlueOfThread = 0
    blueToRedOfThread = 0
    for p in range(1, 13):
        if S.pawn[p-1] != 0:
            if p <= 6:
                nearbyBlueMaxValue = searchNearbyBlueMaxValue(p, S)
                redToBlueOfThread += S.pro[p-1] * nearbyBlueMaxValue
            else:
                nearbyRedMaxValue = searchNearbyRedMaxValue(p, S)
                blueToRedOfThread += S.pro[p-1] * nearbyRedMaxValue
    return redToBlueOfThread, blueToRedOfThread

def findNearby(n, nowPawn):  # 寻找可以移动的棋子
    ans = []
    # 如果有对应棋子
    if nowPawn[n-1] != 0:
        ans.append(n)
    #没有对应棋子
    elif n > 6:
        for i in range(n-1, 6, -1):
            if i in nowPawn:
                ans.append(i)
                break
        for i in range(n+1, 13):
            if i in nowPawn:
                ans.append(i)
                break
    elif n <= 6:
        for i in range(n-1, 0, -1):
            if i in nowPawn:
                ans.append(i)
                break
        for i in range(n+1, 7):
            if i in nowPawn:
                ans.append(i)
                break
    return ans

def getLocation(p, Map):  # 返回传入地图下，棋子p的坐标
    for i in range(5):
        for j in range(5):
            if Map[i][j] == p:
                return i, j

def tryMakeMove(p, PawnMoveTo, S):  # 尝试移动，并且返回移动后的棋局地图与棋子存活情况
    newS = copy.deepcopy(S)
    row, col = getLocation(p, newS.map)
    x = y = 0
    if PawnMoveTo == LEFT:
        y = -1
    elif PawnMoveTo == RIGHT:
        y = +1
    elif PawnMoveTo == UP:
        x = -1
    elif PawnMoveTo == DOWN:
        x = +1
    elif PawnMoveTo == LEFTUP:
        x = -1
        y = -1
    elif PawnMoveTo == RIGHTDOWN:
        x = +1
        y = +1
    # 移动无效
    if notInMap(row+x, col+y):
        return False
    newS.map[row][col] = 0
    row = row + x
    col = col + y
    if newS.map[row][col] != 0:
        i = newS.pawn.index(newS.map[row][col])
        newS.pawn[i] = 0
    newS.map[row][col] = p
    value = getLocValue(newS)  # 获取所有棋子的位置价值
    newS.pro = getPawnPro(newS)  # 获取所有棋子被摇到的概率
    newS.value = getPawnValue(value, newS.pro)
    newS.parent = S
    if S.parent is not None:
        newS.parent_before = S.parent.map
    else:
        newS.parent_before = None
    newS.parent_3 = S.parent_before
    newS.parent_4 = S.parent_3
    newS.pPawn = p
    newS.pMove = PawnMoveTo
    if p < 7:
        newS.cPawn = [INFTY,INFTY,INFTY,INFTY,INFTY,INFTY]
        newS.cPawnSecond = [INFTY,INFTY,INFTY,INFTY,INFTY,INFTY]
    else:
        newS.cPawn = [-INFTY,-INFTY,-INFTY,-INFTY,-INFTY,-INFTY]
        newS.cPawnSecond = [-INFTY,-INFTY,-INFTY,-INFTY,-INFTY,-INFTY]
    return newS

def SoftMax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def getScorered(S, k=2.2, lam=5):  # 计算此时蓝方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]

    theValue = lam * (k * expBlue - expRed) - redToBlueOfThread + blueToRedOfThread
    return theValue

def getDemoValueblue(S, k=2.2, lam=5):
    move = ['right', 'down', 'rightdown']
    exp = 0
    for p in range(1, 7):
        if p in S.pawn:
            theValue = maxValue = -INFTY
            for m in move:
                newStatus = tryMakeMove(p, m, S)
                if newStatus is not False:
                    theValue = getScorered(newStatus)
                    if theValue > maxValue:
                        maxValue = theValue
            exp += S.pro[p - 1] * maxValue
    return exp

def get_alldata_5(myS: Status):
    datanew = []
    NPmap = np.array(myS.map)
    reddata = np.where(NPmap < 7, NPmap, 0)
    bluedata = np.where(NPmap >= 7, NPmap, 0)
    datanew.append(reddata)
    datanew.append(bluedata)
    if myS.parent is not None:
        NPmap = np.array(myS.parent.map)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_before is not None:
        NPmap = np.array(myS.parent_before)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_3 is not None:
        NPmap = np.array(myS.parent_3)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if myS.parent_4 is not None:
        NPmap = np.array(myS.parent_4)
        reddata = np.where(NPmap < 7, NPmap, 0)
        bluedata = np.where(NPmap >= 7, NPmap, 0)
        datanew.append(reddata)
        datanew.append(bluedata)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)
    return datanew

def cal_first_choice(probli, Vsr, nowS, c=9, value_balance=0.1, _playsr=None):
    if _playsr is None:
        _playsr = {}
    U = c * probli / (_playsr.get(nowS, 0) + 1)
    V = Vsr.get(nowS, 0) * value_balance
    C = U + V
    return C

def selectPawn_run_sim_net(S, n=0, _count=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        _count += 1
        if _count % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
        ans = findNearby(n, S.pawn)
    return n, ans, _count

def getTheNextStepStatus_updata_run_sim_net(SL, count):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        i = random.randint(1, 6)
        n, ans, count = selectPawn_run_sim_net(s, i + o, _count=count)
        for p in ans:
            for m in move:
                newStatus = tryMakeMove(p, m, s)
                if newStatus is not False:
                    newStatus.pDice = i
                    NL.append(newStatus)
                del newStatus
    return NL, count

def redByNeuralUCT(ans):
    global playsr
    global winsr
    # global Vsr
    # global recorder
    # global Vsr_all
    # print("red can move ", ans)
    calculation_time = float(15)
    move = ['right', 'down', 'rightdown']
    Vsr = {}
    Vsr_all = {}
    recorder = {}
    Tempt = {}
    SL = []  # 当前局面下合法的后续走法

    datanew = get_alldata_5(S)

    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                Vsr_all[newStatus] = 0.  # 改为 float
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
                del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    playsr[S] = 0
    num_process = 4  # 多进程数目
    pool = mp.Pool(processes=num_process)  # 进程池

    while time.time() - begin < calculation_time:  # time.time() - begin < calculation_time
        inputs_run_sim_net = []  # 多进程输入
        for _ in range(num_process):
            inputs_run_sim_net.append([playsr, Vsr, SL, datanew, games, 1, 30])
        playsr[S] += num_process
        outputs_run_sim_net = pool.map(run_simulation_network, inputs_run_sim_net)  # 多进程输出

        games += num_process
        for output in outputs_run_sim_net:
            Qsr, The_total_choose, k, visited_states, visited_red = output
            The_total_choose2 = SL[0]  # 解决两个 Status 类 S1,S2 的值相同但 S1 != S2
            for state in Vsr_all:
                if state.__str__() == The_total_choose.__str__():
                    The_total_choose2 = state
                    if playsr.get(The_total_choose2, -1) == -1:
                        playsr[The_total_choose2] = 0
                        winsr[The_total_choose2] = 0
                    break

            for move in visited_states:
                if move == The_total_choose:
                    playsr[The_total_choose2] += 1
                else:
                    if playsr.get(move, -1) == -1:  # 激活已选节点在playsr中，winsr中
                        playsr[move] = 0
                        winsr[move] = 0
                    playsr[move] += 1
                if k == 1:
                    if move == The_total_choose:
                        winsr[The_total_choose2] += 1
                    else:
                        winsr[move] += 1
                if move in visited_red:
                    if recorder.get(The_total_choose2, 0) == 0:
                        recorder[The_total_choose2] = 0
                    recorder[The_total_choose2] += 1
                    if k == 1:
                        Qsr[move] = (Qsr.get(move, 0) + 1) / 2
                    if k == 2:
                        Qsr[move] = (Qsr.get(move, 0) - 1) / 2
                    Vsr_all[The_total_choose2] += Qsr.get(move, 0)
                Vsr[The_total_choose2] = Vsr_all.get(The_total_choose2, 0) / recorder.get(The_total_choose2, 1)

        if games % 40 == 0:
            for move in SL:
                Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)
                # file.write(str(Tempt[move]) + ' ')
            # print(Tempt)
            # file.write('\n')
    playsr[S] = 0
    pool.close()
    pool.join()
    # file.close()
    for move in SL:
        Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)

    move_choose = max(Tempt, key=lambda x: Tempt[x])

    for move in SL:
        if move != move_choose and Tempt[move] == Tempt[move_choose] and move.pMove == 'rightdown':
            move_choose = move
    bestp = move_choose.pPawn
    bestm = move_choose.pMove

    for move in SL:
        print(playsr.get(move, 0))
    for move1 in SL:
        print(Tempt.get(move1, 0))
    for move in SL:
        print("the valuenet total is", Vsr.get(move,0))

    print('we have searched for ', games)
    # print('bestp=', bestp, ' bestm=', bestm)

    del Vsr
    del Vsr_all
    del recorder
    del SL
    gc.collect()
    return bestp, bestm

def run_simulation_network(inputs):  # 红方
    # global winsr
    # global playsr
    # global S
    # global Vsr
    # global Vsr_all
    # global recorder
    playsr, Vsr, availables, datanew, games, c, max_actions = inputs
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}

    # 注意网络层数对应的定义
    device = torch.device("cuda")
    # net = ResNet(BasicBlock, [1, 1, 1, 1])
    # net = net.cuda(device)
    # net.load_state_dict(torch.load('0830_1111_Alpha1.pt'))
    # net.eval()

    net = torch.jit.load("Alpha1.pt")

    valuenet = ResNetValue(BasicBlock, [1, 1, 1, 1], 1)
    valuenet = valuenet.cuda(device)
    valuenet.load_state_dict(torch.load('0830_1111_Alpha1_value_little.pt'))
    valuenet.eval()

    temperature = 0.1
    temperature_demo = 2.0
    count = 0  # 同 COUNT

    Qsr = {}
    # availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    visited_red = set()
    # playsr[S] += 1
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)

    # torch.onnx.export(net, datanew, 'model.onnx', input_names=["input"], output_names=["output"], dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

    # # Tracing
    # traced_net = torch.jit.trace(net, datanew)
    #
    # # 保存Torch Script模型
    # traced_net.save("Alpha1.pt")

    # init jit
    # for _ in range(10):
    #     run_model(net, datanew)
    #     run_model(net_jit, datanew)
    #
    # test_times = 10
    #
    # # begin testing
    # results = pd.DataFrame({
    #     "type": ["orgin"] * test_times + ["jit"] * test_times,
    #     "cost_time": [run_model(net, datanew) for _ in range(test_times)] + [run_model(net_jit, datanew) for _ in
    #                                                                                 range(test_times)]
    # })
    #
    # plt.figure(dpi=120)
    # sns.boxplot(
    #     x=results["type"],
    #     y=results["cost_time"]
    # )
    # plt.show()
    #
    # results.to_csv("test_results0.csv", index=False)
    #
    # print(results)

    outnew = net(datanew)

    prenew = outnew
    out = torch.zeros(25)
    prenew = prenew.squeeze()
    # out = self.trans(out)
    out = out.cuda(device)
    # print(out)
    for i in range(0, 6):
        oneStatus = prenew[i * 4:(i + 1) * 4]
        out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
        for index, value in enumerate(out[i * 4:(i + 1) * 4]):
            if index < 3:
                if value < 0.2:
                    out[i * 4 + index] = 0.2
    out = out.tolist()
    # probli = []
    consider = {}

    for move in availables:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        # if games == 0:
        #     print(move.pMove, probli)
        C = cal_first_choice(probli, Vsr, move, 9, _playsr=playsr)
        consider[move] = C
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    del consider
    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    # if playsr.get(The_total_choose, -1) == -1:
    #     playsr[The_total_choose] = 0
    #     winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):

                NL = []
                NL.append(move_choose)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL
                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            move_otherside = move
                            break
                if k:
                    break

                max_problis = []
                consider = {}
                max_consider = []
                for move in availables:
                    theValue = getDemoValueblue(move)
                    consider[move] = theValue
                    max_problis.append(theValue)
                    max_consider.append(move)

                max_problis = np.array(max_problis)
                move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature_demo), k=1)
                move_otherside = move_otherside[0]

                visited_states.add(move_otherside)
                # if playsr.get(move_otherside, -1) == -1:
                #     playsr[move_otherside] = 0
                #     winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                NL = []
                NL.append(move_otherside)
                availables, count = getTheNextStepStatus_updata_run_sim_net(NL, count)
                del NL

                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            move_choose = move
                            break
                if k:
                    break

                datanew = get_alldata_5(move_otherside)
                datanew = np.array(datanew)
                datanew = torch.tensor(datanew)
                datanew = datanew.cuda(device)
                datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
                outnew = net(datanew)
                # # Tracing
                # traced_net = torch.jit.trace(net, datanew)
                # # 保存Torch Script模型
                # traced_net.save("Alpha1_CPU.pt")

                prenew = outnew
                out = torch.zeros(25)
                prenew = prenew.squeeze()
                # out = self.trans(out)
                out = out.cuda(device)
                # print(out)
                for i in range(0, 6):
                    oneStatus = prenew[i * 4:(i + 1) * 4]
                    out[i * 4:(i + 1) * 4] = 3 * torch.softmax(oneStatus, 0)
                    for index, value in enumerate(out[i * 4:(i + 1) * 4]):
                        if index < 3:
                            if value < 0.2:
                                out[i * 4 + index] = 0.2
                out = out.tolist()
                # probli = []
                consider = {}
                max_consider = []
                max_problis = []

                for move in availables:
                    move_p = move.pPawn
                    move_to = move.pMove
                    problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
                    probli = problis[move_dict[move_to]]
                    max_problis.append(probli)
                    consider[move] = probli
                list_consider_value = list(consider.values())
                list_consider_keys = list(consider.keys())
                for one_pro in max_problis:
                    position = list_consider_value.index(one_pro)
                    max_consider.append(list_consider_keys[position])
                max_problis = torch.tensor(max_problis)

                move_choose = random.choices(max_consider, torch.softmax(max_problis / temperature, 0), k=1)

                move_choose = move_choose[0]

                Qsr[move_choose] = valuenet(datanew).tolist()[0][0]
                # Qsr[move_choose] = 0
                visited_red.add(move_choose)

                # 激活已选节点在playsr中，winsr中
                # if playsr.get(move_choose, -1) == -1:
                #     playsr[move_choose] = 0
                #     winsr[move_choose] = 0
                visited_states.add(move_choose)
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    # for move in visited_states:
    #     playsr[move] += 1
    #     if k == 1:
    #         winsr[move] += 1
    #     if move in visited_red:
    #         if recorder.get(The_total_choose, 0) == 0:
    #             recorder[The_total_choose] = 0
    #         recorder[The_total_choose] += 1
    #         if k == 1:
    #             Qsr[move] = (Qsr.get(move, 0) + 1) / 2
    #         if k == 2:
    #             Qsr[move] = (Qsr.get(move, 0) - 1) / 2
    #         Vsr_all[The_total_choose] += Qsr.get(move, 0)
    #     Vsr[The_total_choose] = Vsr_all.get(The_total_choose, 0) / recorder.get(The_total_choose, 1)
    # del visited_states
    # del visited_red
    return Qsr, The_total_choose, k, visited_states, visited_red

# 20190302 socket 移动棋子,如果走错直接输
def socketToMove(conn,n, ans, S):
    message = str(S.map) + '|' + str(n)
    conn.sendall(message.encode('UTF-8'))
    try:
        conn.settimeout(1500)
        data, address = conn.recvfrom(1024)
    except socket.error as e:
        logger.info(str(e))
        return -1, 'timeout'

    # 对战
    text = (data.decode()[:-1]).split('|')
    # 调试
    # text = (data.decode()).split('|')
    p = int(text[0])
    moveTo = text[1]
    if (p in ans):
        if p > 0 and p < 7:
            if moveTo == DOWN or moveTo == RIGHT or moveTo == RIGHTDOWN:
                newS = tryMakeMove(p, moveTo, S)
                if newS is not False:
                    return p, moveTo
        elif p > 6 and p < 13:
            if moveTo == UP or moveTo == LEFT or moveTo == LEFTUP:
                newS = tryMakeMove(p, moveTo, S)
                if newS is not False:
                    return p, moveTo
    return -1, 'not move'

def playGame(Red, Blue, detail, conn):
    lastInfo = []
    mapNeedRedraw = True  # 是否需要重新绘制地图
    movered_moveblue = {'right': 'left', 'down': 'up', 'rightdown': 'leftup'}

    if detail:
        drawGameScreen(Red, Blue)
    while True:
        moveTo = None  # 棋子移动方向
        mapNeedRedraw = False

        # correct = 0 #游戏结果

        n, ans = selectPawn(S)
        if detail:
            drawMovePawn(n, ans)

        # GUI control here
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()

        if COUNT % 2 == 0:
            if Red == 'Nerual_Uct':
                p, moveTo = redByNeuralUCT(ans)

        if COUNT % 2 == 1:
            if Blue == 'Socket':
                try:
                    p, moveTo = socketToMove(conn=conn, n=n, ans=ans, S=S)
                    if p == -1:
                        logger.info('RESULT : REDWIN')
                        return REDWIN
                except socket.error as e:
                    logger.info('RESULT : REDWIN')
                    return REDWIN
                except ValueError as e1:
                    logger.info('RESULT : REDWIN')
                    return REDWIN

        if moveTo != None:
            moved = makeMove(p, moveTo)
            lastInfo = [n, p, moveTo]
            if moved:
                mapNeedRedraw = True
        if mapNeedRedraw and detail:  # 如果需要重绘棋局界面，则：
            sleep(SLEEPTIME)
            drawGameScreen(Red, Blue)  # 重绘棋局界面
            logger.info(str(S.map)+' |chess: '+str(p)+' |move : '+moveTo)
            pass

        result = isEnd(S)  # 检查游戏是否结束，返回游戏结果

        if result:
            lastInfo = []
            logger.info('RESULT : REDWIN' if result == 1 else 'RESULT : ' + ID + 'WIN')
            return result

def startGame(Red, Blue, n, filename, detail=True):
    global COUNT
    global S
    global playsr
    global winsr
    init()
    if detail:
        drawStartScreen()  # 游戏开始界面
    RESULT[0] = 0
    RESULT[1] = 0
    cnt = n
    rateline = []
    if Blue == 'Socket' or Red == 'Socket':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', 50007))
        sock.listen(1)
        conn, addr = sock.accept()
        global ID
        ID = conn.recv(1024).decode()[:-1]
        print("CONNECT CLIENT ID : " + str(ID))
    else:
        conn = 0
    print('   connected')
    while cnt:
        global S
        S = Status()
        resetInfo()
        playsr = {}
        winsr = {}
        result = playGame(Red, Blue, detail, conn)  # 游戏开始，返回比赛结果
        gc.collect()
        del S
        del winsr
        del playsr
        if detail:
            #pass
            drawWinScreen(result)

        RESULT[result-1] += 1                       # 更新比分
        cnt -= 1
        COUNT = 2000 - cnt - 1                      # 先手方交替
        rateline.append(float(RESULT[0])/sum(RESULT))

        drawText('RED : ' + str(RESULT[0]), font, RED, screen, 6, 7)
        drawText('BLUE : ' + str(RESULT[1]), font, BLUE, screen, 6.5, 7)
        pygame.display.update()

        if cnt % 5 == 0:
            logger.info(f"{sum(RESULT)}\t{round(100*RESULT[0]/sum(RESULT),4)}")

    if Blue == 'Socket' or Red == 'Socket':
        try:
            conn.sendall('close'.encode('utf8'))
            conn.close()
        except socket.error as e:
            print('Test Game Over : %s' % e)
    return RESULT[0]


if __name__ == '__main__':
    '''
    可选测试对象
    Red  ：BetaCat1.0 | Socket | Uct
    Blue ：Demo | Socket '
    BetaCat1.0 Demo 分别为红方或蓝方AI
    Socket 为用户选择红方或蓝方连接客户端，不能同时连接客户端
    当前示例表示红方为服务器，蓝方为客户端
    首先启动此程序
    '''
    mp.set_start_method('spawn')
    Red = 'Nerual_Uct'
    Blue = 'Socket'
    filename = os.getcwd() + "\\data\\" + Red + 'Vs' + Blue
    print(filename)
    # 测试局数
    cnt = 100
    # temperature = 0.1
    result = startGame(Red, Blue, cnt, filename, detail=True)
    input('wait')
