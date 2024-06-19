# from typing_extensions import get_origin
import pygame
import os
import torch
import sys
import random
import copy
import time
import gc
import math
import numpy as np
from torch import multiprocessing as mp
from random import choice, shuffle
from math import log, sqrt
from pygame.locals import *
from torch import softmax
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

from try_resnet_0706 import ResNet
from try_resnet_0706 import BasicBlock
from try_resnet_0713_value import ResNet as ResNetValue

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
torch.backends.cudnn.benchmark = True

ti = time.time()
# tr = 0
# tb = 0
WINDOWSIZE = (1100, 680)  # 游戏窗口大小
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
TIPSIZE = 12  # 提示文字大小
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
i = 1
myTime = 0    # 己方所用时间
yourTime = 0  # 对方所用时间


class Status(object):
    def __init__(self):
        self.map = None  # 矩阵棋盘
        self.value = None  # 所有棋子的价值
        self.pawn = None  # 棋子列表，没有的标记为0
        self.pro = None  # 所有棋子被摇到的概率
        self.parent_before = None
        self.parent = None  # 此局面上一轮时状态
        self.parent_3 = None
        self.parent_4 = None

        self.pPawn = None  # 上一轮选择的棋子
        self.pMove = None  # 上一轮选择的移动方向
        self.pDice = None  # 父节点的骰子数
        self.cPawn = None
        self.cPawnSecond = None
        self.indx = None  # 记录局面所处的步数
        self.cPM = [[], [], [], [], [], []]
        self.cPMSecond = [[], [], [], [], [], []]
        self.children = []

    def print(self):
        print(self.cPM)

    def __str__(self):
        # print(Status)
        return '[棋子为%s , 选择方向为%s]' % (self.pPawn, self.pMove)


def init():
    global IMAGE, tip, screen, font, maplib, Lyr, Lyb, Lx, S, matchPro
    pygame.init()
    S = Status()

    # 布局库
    maplib = [[6, 2, 4, 1, 5, 3],
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
    # resetInfo()  # 重置比赛信息  # Annotated in 2023-8-4
    Lyr = []
    Lyb = []
    Lx = []
    matchPro = 0.85


def loadImage(name, pos, size=SIZE):  # 放置图片
    filename = "picture/white/" + name
    screen.blit(pygame.transform.scale(  # blit 把一张图A粘贴到另一张图B上
        pygame.image.load(filename).convert(), size), pos)


def showWinRate(RedWinRate, BlueWinRate, x):  # 可视化胜率
    global Lyr, Lyb, Lx
    yr = (100 - RedWinRate) / (100 / 3.0) + 0.6
    yb = (100 - BlueWinRate) / (100 / 3.0) + 0.6
    x = x / (1000 / 5) + 4.2
    Lyr.append(copy.deepcopy(yr))
    Lyb.append(copy.deepcopy(yb))
    Lx.append(copy.deepcopy(x))
    for i in range(0, len(Lyr) - 1):
        pygame.draw.line(
            screen, RED, (100 * Lx[i], 100 * Lyr[i]), (100 * Lx[i], 100 * Lyr[i + 1]))
        pygame.draw.line(
            screen, BLUE, (100 * Lx[i], 100 * Lyb[i]), (100 * Lx[i], 100 * Lyb[i + 1]))


def drawGameScreen(Red, Blue):  # 游戏比赛界面
    global S
    S.pro = getPawnPro(S)
    locValue = getLocValue(S)
    S.value = getPawnValue(S.pro, locValue)
    screen.fill(WHITE)
    # 画棋盘
    for i in range(6):
        x = y = 60 * (i + 1)
        pygame.draw.line(screen, LINECOLOR, (60, y), (360, y))
        pygame.draw.line(screen, LINECOLOR, (x, 60), (x, 360))
    # 加载提示文字
    drawText('Winning Percentage Dynamic Figure', font, BLACK, screen, 0, 7.2)
    drawText(Red + ' Vs ' + Blue, font, BLACK, screen, 0.5, 7.2)
    drawText('matchPro : ' + str(round(100 * matchPro, 4)) + '%', font, BLACK, screen, 1, 7.2)
    pygame.draw.line(screen, LINECOLOR, (415, 55), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (425, 55), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (420, 360), (420, 50))
    pygame.draw.line(screen, LINECOLOR, (420, 360), (780, 360))
    pygame.draw.line(screen, LINECOLOR, (775, 355), (780, 360))
    pygame.draw.line(screen, LINECOLOR, (775, 365), (780, 360))
    drawText('You can move: ', font, BLACK, screen, 0.1, 0.2)
    drawText('A : down', tip, BLACK, screen, 6, 1)
    drawText('W : right', tip, BLACK, screen, 6, 2.5)
    drawText('S : right-down', tip, BLACK, screen, 6, 4)
    drawText('U : up', tip, BLACK, screen, 6.5, 1)
    drawText('H : left', tip, BLACK, screen, 6.5, 2.5)
    drawText('Y : left-up', tip, BLACK, screen, 6.5, 4)
    drawText('RED : ' + str(RESULT[0]), font, RED, screen, 6, 7)
    drawText('BLUE : ' + str(RESULT[1]), font, BLUE, screen, 6.5, 7)
    if (sum(RESULT)):
        RedWinRate = round(100 * float(RESULT[0]) / sum(RESULT), 2)
        BlueWinRate = round(100 * float(RESULT[1]) / sum(RESULT), 2)
        drawText('RedWinRate:' + str(round(100 *
                                           float(RESULT[0]) / sum(RESULT), 2)), font, RED, screen, 6, 9.5)
        drawText('BlueWinRate:' + str(round(100 *
                                            float(RESULT[1]) / sum(RESULT), 2)), font, BLUE, screen, 6.5, 9.5)
        x = sum(RESULT)
        showWinRate(RedWinRate, BlueWinRate, x)
    # 画棋子
    for i in range(5):
        for j in range(5):
            if S.map[i][j] != 0:
                drawPawn(S.map[i][j], i, j)
    drawPawnProAndValue()
    drawTime()
    pygame.display.update()


def drawMovePawn(n, ans):  # 可选择移动的棋子
    x = -1
    y = 2
    for v in ans:
        drawPawn(v, x, y)
        y += 1
    if n <= 6:
        loadImage(str(n) + '.png', (310, 5))
    else:
        loadImage(str(n - 6) + '.png', (310, 5))
    pygame.display.update()


def drawPawn(value, row, col, size=SIZE):  # 在（row，col）处，画值为value的棋子
    pos_x = col * STEP + START
    pos_y = row * STEP + START
    Pos = (pos_x, pos_y)
    if value <= 6:
        s = 'R' + str(value)
    elif value > 6:
        s = 'B' + str(value - 6)
    loadImage(s + '.png', Pos, size)


def drawPawn1(value, row, col, size=SIZE):  # 在（row，col）处，画值为value的棋子（此函数与上一函数几乎相同，只是改了一个参数，为了布局美化而已）
    pos_x = col * STEP + START
    pos_y = row * STEP + 1.2 * START  # 只改了这里，为了调位置
    Pos = (pos_x, pos_y)
    if value <= 6:
        s = 'R' + str(value)
    elif value > 6:
        s = 'B' + str(value - 6)
    loadImage(s + '.png', Pos, size)


def drawText(text, font, color, surface, row, col):  # 处理需要描绘的文字：text：文本；font：格式；
    row += 0.2
    x = col * STEP
    y = row * STEP
    textobj = font.render(text, True, color, WHITE)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def drawTime():  # 显示每一方累计花费时间
    pass
    # global ti
    # drawText('RedTime:', font, BLACK, screen, 9, 13.5)
    # drawText('{}'.format(round(tr, 2)), font, BLACK, screen, 9, 16)
    # drawText('BlueTime:', font, BLACK, screen, 10, 13.5)
    # drawText('{}'.format(round(tb, 2)), font, BLACK, screen, 10, 16)


def drawPawnProAndValue():  # 在主界面上显示文字与图片
    pro = getPawnPro(S)
    value = getLocValue(S)
    value = getPawnValue(pro, value)
    drawText('Value', font, BLACK, screen, 7.1, 0.2)
    drawText('Pro', font, BLACK, screen, 8.1, 0.3)
    drawText('Value', font, BLACK, screen, 9.1, 0.2)
    drawText('Pro', font, BLACK, screen, 10.1, 0.3)
    drawText('RedScore:', font, BLACK, screen, 7.1, 13.5)
    drawText('{}'.format(round(getScore(S), 2)), font, BLACK, screen, 7.1, 16)
    drawText('BlueScore:', font, BLACK, screen, 8.1, 13.5)
    drawText('{}'.format(round(getDemoValue(S), 2)), font, BLACK, screen, 8.1, 16)

    for p in range(1, 13):
        if p < 7:
            drawPawn1(p, 6, 2 * (p - 1) + 0.3, (30, 30))
            drawPawn1(p, 7, 2 * (p - 1) + 0.3, (30, 30))
            drawText(str(round(value[p - 1], 2)),
                     font, BLACK, screen, 7.1, 2 * p + 0.1)
            drawText(str(round(pro[p - 1], 2)), font, BLACK, screen, 8.1, 2 * p + 0.1)
        else:
            drawPawn1(p, 8, 2 * (p - 7) + 0.3, (30, 30))
            drawPawn1(p, 9, 2 * (p - 7) + 0.3, (30, 30))
            drawText(str(round(value[p - 1], 2)), font,
                     BLACK, screen, 9.1, 2 * (p - 6) + 0.1)
            drawText(str(round(pro[p - 1], 2)), font,
                     BLACK, screen, 10.1, 2 * (p - 6) + 0.1)


def selectPawn(S, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    global COUNT
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        COUNT += 1
        if COUNT % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
        ans = findNearby(n, S.pawn)
    return n, ans


def selectPawnnewone(S, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    global COUNT
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        COUNT += 1
        if COUNT % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
        COUNT += 1
        ans = findNearby(n, S.pawn)
    return n, ans


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


def selectPawnnew(S, n=0):  # 掷骰子，挑选可以移动的棋子。n为骰子数，0表示为了模拟
    global COUNTNEW
    if n == 0:  # 未传入n说明不是根据现局面与骰子模拟，在模拟后面棋局
        COUNTNEW += 1
        if COUNTNEW % 2 == 0:  # 偶数是红，奇数是蓝
            n = random.randint(1, 6)  # 红
        else:
            n = random.randint(7, 12)  # 蓝
        ans = findNearby(n, S.pawn)
    else:  # 如果传入了n，说明是为了模拟真实棋局
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
    if notInMap(row + x, col + y):
        return False
    S.map[row][col] = 0
    row = row + x
    col = col + y
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


def showSelected(p):  # 用红色标记，显示被挑选的棋子
    row, col = getLocation(p, S.map)
    pos_x = col * STEP + START
    pos_y = row * STEP + START
    Pos = (pos_x, pos_y)
    if p > 6:
        s = 'Y' + str(p - 6)
    else:
        s = 'Y' + str(p)
    loadImage(s + '.png', Pos)
    # screen.blit(IMAGE[s],Pos)
    pygame.display.update()


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
    # global tr, tb
    # tr = tb = 0
    # S.map = getNewMap()
    S.map = getcommap()
    S.pawn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 棋子初始化
    S.pro = [1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 /
             6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]
    # value = getLocValue(S)
    # S.value = getPawnValue(S.pro, value)


def getcommap():
    global remymap
    if COUNT % 2 == 1:
        a1 = input('输入对手棋1: ')
        a2 = input('输入对手棋2: ')
        a3 = input('输入对手棋3: ')
        a4 = input('输入对手棋4: ')
        a5 = input('输入对手棋5: ')
        a6 = input('输入对手棋6: ')

        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)
        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a1 + 6],
            [0, 0, 0, a2 + 6, a3 + 6],
            [0, 0, a4 + 6, a5 + 6, a6 + 6]
        ]
    else:
        a1 = input('输入对手棋1: ')
        a2 = input('输入对手棋2: ')
        a3 = input('输入对手棋3: ')
        a4 = input('输入对手棋4: ')
        a5 = input('输入对手棋5: ')
        a6 = input('输入对手棋6: ')

        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)

        newMap = [
            [6, 2, 4, 0, 0],
            [1, 5, 0, 0, 0],
            [3, 0, 0, 0, a6 + 6],
            [0, 0, 0, a5 + 6, a4 + 6],
            [0, 0, a3 + 6, a2 + 6, a1 + 6]
        ]
    remymap = [a1, a2, a3, a4, a5, a6]
    return newMap


def getNewMap():  # 换新图
    r = random.sample(maplib, 1)[0]
    b = random.sample(maplib, 1)[0]
    newMap = [
        [6, 2, 4, 0, 0],
        [1, 5, 0, 0, 0],
        [3, 0, 0, 0, b[2] + 6],
        [0, 0, 0, b[4] + 6, b[1] + 6],
        [0, 0, b[5] + 6, b[3] + 6, b[0] + 6]
    ]
    return newMap


def getLocValue(S):  # 棋子所在位置的价值
    blueValue = [[99, 10, 6, 3, 1],
                 [10, 8, 4, 2, 1],
                 [6, 4, 4, 2, 1],
                 [3, 2, 2, 2, 1],
                 [1, 1, 1, 1, 1]]
    redValue = [[1, 1, 1, 1, 1],
                [1, 2, 2, 2, 3],
                [1, 2, 4, 4, 6],
                [1, 2, 4, 8, 10],
                [1, 3, 6, 10, 99]]
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(1, 13):
        if S.pawn[p - 1] != 0:
            row, col = getLocation(p, S.map)
            if p <= 6:
                V[p - 1] = redValue[row][col]
            else:
                V[p - 1] = blueValue[row][col]
    return V


def getPawnValue(value, pro):  # 棋子价值
    V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 12):
        V[i] = pro[i] * value[i]
    return V


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


def getPawnProred(S):  # 返回棋子被摇到的概率
    pro = [0, 0, 0, 0, 0, 0]
    for p in range(1, 7):
        if S.pawn[p - 1] != 0:
            pro[p - 1] = pro[p - 1] + 1
        if S.pawn[p - 1] == 0:
            pro[p - 1] = 0
            ans = findNearby(p, S.pawn)
            if len(ans) > 1:
                pr = ans[0] - 1
                pl = ans[1] - 1
                pro[pr] = pro[pr] + 1
                pro[pl] = pro[pl] + 1

            elif len(ans) == 1:
                pro[ans[0] - 1] += 1
            elif len(ans) == 0:
                pass
    for p in range(1, 7):
        pro[p - 1] = pro[p - 1] / 6
    return pro


def getPawnProblue(S):  # 返回棋子被摇到的概率
    pro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for p in range(7, 13):
        if S.pawn[p - 1] != 0:
            pro[p - 1] = pro[p - 1] + 1
        if S.pawn[p - 1] == 0:
            pro[p - 1] = 0
            ans = findNearby(p, S.pawn)
            if len(ans) > 1:
                pr = ans[0] - 1
                pl = ans[1] - 1
                pro[pr] = pro[pr] + 1
                pro[pl] = pro[pl] + 1

            elif len(ans) == 1:
                pro[ans[0] - 1] += 1
            elif len(ans) == 0:
                pass
    for p in range(7, 13):
        pro[p - 1] = pro[p - 1] / 6
    return pro


def searchNearbyBlueMaxValue(p, S):  # 搜索附近蓝方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row + 1 < 5:
        if S.map[row + 1][col] > 6:
            nearby.append(S.value[S.map[row + 1][col] - 1])
    if col + 1 < 5:
        if S.map[row][col + 1] > 6:
            nearby.append(S.value[S.map[row][col + 1] - 1])
    if row + 1 < 5 and col + 1 < 5:
        if S.map[row + 1][col + 1] > 6:
            nearby.append(S.value[S.map[row + 1][col + 1] - 1])
    if nearby == []:
        return 0

    expValue = 0
    for v in nearby:
        expValue += v / sum(nearby)
    # print("the expvalue is",expValue)
    return expValue


def searchNearbyRedMaxValue(p, S):  # 搜索附近红方最有价值的棋子
    nearby = []
    row, col = getLocation(p, S.map)
    if row - 1 >= 0:
        if S.map[row - 1][col] <= 6 and S.map[row - 1][col] > 0:
            nearby.append(S.value[S.map[row - 1][col] - 1])
    if col - 1 >= 0:
        if S.map[row][col - 1] <= 6 and S.map[row][col - 1] > 0:
            nearby.append(S.value[S.map[row][col - 1] - 1])
    if row - 1 >= 0 and col - 1 >= 0:
        if S.map[row - 1][col - 1] <= 6 and S.map[row - 1][col - 1] > 0:
            nearby.append(S.value[S.map[row - 1][col - 1] - 1])
    if nearby == []:
        return 0
    expValue = 0
    for v in nearby:
        expValue += v / sum(nearby)

    return expValue


def getThread(S):  # 获得红方对蓝方的威胁值，蓝方对红方的威胁值
    redToBlueOfThread = 0
    blueToRedOfThread = 0
    for p in range(1, 13):
        if S.pawn[p - 1] != 0:
            if p <= 6:
                nearbyBlueMaxValue = searchNearbyBlueMaxValue(p, S)
                redToBlueOfThread += S.pro[p - 1] * nearbyBlueMaxValue
            else:
                nearbyRedMaxValue = searchNearbyRedMaxValue(p, S)
                blueToRedOfThread += S.pro[p - 1] * nearbyRedMaxValue
    return redToBlueOfThread, blueToRedOfThread


def findNearby(n, nowPawn):  # 寻找可以移动的棋子，n是当前骰子数，返回所有符合条件棋子
    ans = []
    if nowPawn[n - 1] != 0:
        ans.append(n)
    elif n > 6:
        for i in range(n - 1, 6, -1):
            if i in nowPawn:
                ans.append(i)
                break
        for i in range(n + 1, 13):
            if i in nowPawn:
                ans.append(i)
                break
    elif n <= 6:
        for i in range(n - 1, 0, -1):
            if i in nowPawn:
                ans.append(i)
                break
        for i in range(n + 1, 7):
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
    if notInMap(row + x, col + y):
        return False
    newS.map[row][col] = 0
    row = row + x
    col = col + y
    if newS.map[row][col] != 0:  # 检查移动的目标格子位是否有棋子，是的话被吃掉，赋值为0
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
        newS.cPawn = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
        newS.cPawnSecond = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
    else:
        newS.cPawn = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
        newS.cPawnSecond = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
    return newS


def decideRedHowToMove(ans):  # 人类选手决定如何行棋
    p = 0
    while True:
        PawnMoveTo = None
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            elif event.type == KEYDOWN:
                if len(ans) > 1:
                    if event.key == K_1:
                        p = 1
                    elif event.key == K_2:
                        p = 2
                    elif event.key == K_3:
                        p = 3
                    elif event.key == K_4:
                        p = 4
                    elif event.key == K_5:
                        p = 5
                    elif event.key == K_6:
                        p = 6
                else:
                    p = ans[0]
                if p != 0:
                    if p in ans:
                        showSelected(p)
                        if event.key == K_w:
                            PawnMoveTo = RIGHT
                        elif event.key == K_a:
                            PawnMoveTo = DOWN
                        elif event.key == K_s:
                            PawnMoveTo = RIGHTDOWN
        if PawnMoveTo != None:
            newS = tryMakeMove(p, PawnMoveTo, S)
            if newS is not False:
                return p, PawnMoveTo


def decideBlueHowToMove(ans):
    p = 0
    while True:
        PawnMoveTo = None
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            elif event.type == KEYDOWN:
                if len(ans) > 1:
                    if event.key == K_1:
                        p = 7
                    elif event.key == K_2:
                        p = 8
                    elif event.key == K_3:
                        p = 9
                    elif event.key == K_4:
                        p = 10
                    elif event.key == K_5:
                        p = 11
                    elif event.key == K_6:
                        p = 12
                else:
                    p = ans[0]
                if p != 0:
                    if p in ans:
                        showSelected(p)
                        if event.key == K_h:
                            PawnMoveTo = LEFT
                        elif event.key == K_u:
                            PawnMoveTo = UP
                        elif event.key == K_y:
                            PawnMoveTo = LEFTUP
        if PawnMoveTo != None:
            newS = tryMakeMove(p, PawnMoveTo, S)
            if newS is not False:
                return p, PawnMoveTo


def checkChess(chess: int) -> bool:
    mp = np.array(S.map)
    # print(mp)
    for i in range(5):
        for j in range(5):
            if  chess + 6 == mp[i][j]:
                return True
    return False


def decideBlueHowToMove2(ans):
    global yourTime
    twice = ''  # 判断是否输入错误
    if playhand == 'first':
        for i in range(5):
            mychoice = input('请'+twice+'输入要移动的棋子：')
            yourTime += time.time() - tmpTime  # 计算对方行棋时间
            if not checkChess(int(mychoice)):
                print("!输入错误!")
                twice = '重新'
                continue
            mymove = input('移动方向q(leftup) w(up) a(left)：')
            if mymove != 'q' and mymove != 'a' and mymove != 'w':
                print("!输入错误!")
                twice = '重新'
                continue
            if mymove == 'q':
                PawnMoveTo = 'leftup'
            if mymove == 'a':
                PawnMoveTo = 'left'
            if mymove == 'w':
                PawnMoveTo = 'up'

            p = int(mychoice)
            p = p + 6
            if p > 6 and p < 13:
                break
    else:
        for i in range(5):
            mychoice = input('请'+twice+'输入要移动的棋子：')
            yourTime += time.time() - tmpTime  # 计算对方行棋时间
            if not checkChess(int(mychoice)):
                print("!输入错误!")
                twice = '重新'
                continue
            mymove = input('移动方向o(right) k(down) l(rightdown)：')
            if mymove != 'l' and mymove != 'k' and mymove != 'o':
                print("!输入错误!")
                twice = '重新'
                continue
            if mymove == 'l':
                PawnMoveTo = 'leftup'
            if mymove == 'k':
                PawnMoveTo = 'up'
            if mymove == 'o':
                PawnMoveTo = 'left'

            p = int(mychoice)
            p = p + 6
            if p > 6 and p < 13:
                break
    return p, PawnMoveTo


def blueByBraveOfMan(ans):
    moveTo = ['leftup', 'up', 'left']
    bestp = 0
    bestm = ''
    for p in ans:
        for m in moveTo:
            newS = tryMakeMove(p, m, S)
            if newS is not False:
                bestp = p
                bestm = m
                del newS
                break

    return bestp, bestm


def redByBraveOfMan(ans):
    moveTo = ['rightdown', 'down', 'right']
    bestp = 0
    bestm = ''
    for p in ans:
        for m in moveTo:
            newS = tryMakeMove(p, m, S)
            if newS is not False:
                bestp = p
                bestm = m
                break
    return bestp, bestm


def getScore(S, k=2.2, lam=5):  # 计算此时红方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]
    theValue = lam * (k * expRed - expBlue) - blueToRedOfThread + redToBlueOfThread
    return theValue


def getScoreblue(S, k=2.2, lam=5):  # 计算此时蓝方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]
    theValue = lam * (k * expRed - expBlue) + redToBlueOfThread - blueToRedOfThread
    return theValue


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


def blueByDemo(ans, k=2.2, lam=5):  # 一种简单的蓝方走子策略
    maxValue = theValue = -INFTY
    bestp = 0
    bestm = ''
    move = ['left', 'up', 'leftup']
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                theValue = getDemoValue(newStatus)
                if theValue > maxValue:
                    maxValue, bestp, bestm = theValue, p, m
    print(bestp, bestm)
    return bestp, bestm


def redByDemo(ans, k=2.2, lam=5):  # 一种简单的蓝方走子策略
    maxValue = theValue = -INFTY
    bestp = 0
    bestm = ''
    move = ['left', 'up', 'leftup']
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                theValue = getDemoValue(newStatus)
                if theValue > maxValue:
                    maxValue, bestp, bestm = theValue, p, m
    print(bestp, bestm)
    return bestp, bestm


def blueByDemo2(ans, k=2.2, lam=5):  # 一种简单的蓝方走子策略
    maxValue = theValue = -INFTY
    bestp = 0
    bestm = ''
    move = ['left', 'up', 'leftup']
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                theValue = getDemoValue2(newStatus)
                if theValue > maxValue:
                    maxValue, bestp, bestm = theValue, p, m
    return bestp, bestm


def getDemoValue(S, x=0, k=2.2, lam=5):  # 此时蓝方的局面估值
    redToBlueOfThread, blueToRedOfThread = getThread(S)
    expRed = expBlue = 0
    for i in range(0, 12):
        if i < 6:
            expRed += S.value[i]
        else:
            expBlue += S.value[i]
    theValue = lam * (k * expBlue - expRed) + blueToRedOfThread - redToBlueOfThread
    if x == 1:
        print(expBlue, expRed, blueToRedOfThread, redToBlueOfThread)
    return theValue


def getDemoValue2(S, k=2.2, lam=5):
    move = ['right', 'down', 'rightdown']
    exp = 0
    for p in range(1, 7):
        if p in S.pawn:
            theValue = maxValue = -INFTY
            for m in move:
                newStatus = tryMakeMove(p, m, S)
                if newStatus is not False:
                    theValue = getScore(newStatus, 0.4794, 2.1756)
                    if theValue > maxValue:
                        maxValue = theValue
            exp += S.pro[p - 1] * maxValue
    return -exp


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


def getDemoValuered(S, k=2.2, lam=5):
    move = ['left', 'up', 'leftup']
    exp = 0
    for p in range(7, 12):
        if p in S.pawn:
            theValue = maxValue = -INFTY
            for m in move:
                newStatus = tryMakeMove(p, m, S)
                if newStatus is not False:
                    theValue = getScoreblue(newStatus)
                    if theValue > maxValue:
                        maxValue = theValue
            exp += S.pro[p - 1] * maxValue
    return exp


def redByMinimax(ans, k=2.2, lam=5, STEP=2):  # 期望极大极小
    maxValue = theValue = -INFTY
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
    STEP -= 1
    if len(SL) == 1:
        bestp, bestm = SL[0].pPawn, SL[0].pMove
    else:
        KL.append(SL)
        for i in range(STEP):
            NL = getTheNextStepStatus(KL[-1])
            KL.append(NL)
        KL = MinimaxGoBack(KL)
        for s in KL:
            theValue = getSum(s.cPawn)
            if theValue > maxValue:
                maxValue, bestp, bestm = theValue, s.pPawn, s.pMove
    print('the fast without UCT result is ', bestp, bestm)
    # return bestp, bestm


def blueByMinimax(ans, k=2.2, lam=5, STEP=2):  # 期望极大极小
    maxValue = theValue = -INFTY
    bestp = 0;
    bestm = '';
    move = ['left', 'up', 'leftup']
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
    STEP -= 1
    if len(SL) == 1:
        bestp, bestm = SL[0].pPawn, SL[0].pMove
    else:
        KL.append(SL)
        for i in range(STEP):
            NL = getTheNextStepStatus(KL[-1])
            KL.append(NL)
        KL = MinimaxGoBack1(KL)
        for s in KL:
            theValue = getSum(s.cPawn)
            if theValue > maxValue:
                maxValue, bestp, bestm = theValue, s.pPawn, s.pMove
    return bestp, bestm


def MinimaxGoBack1(KL):  # 和MinimaxGoBack一样，只不过1是蓝方期望极大极小，没有1的是红方期望搜素。。。
    for s in KL[-1]:
        score = getDemoValue(s)
        if s.pPawn <= 6:
            if score > s.parent.cPawn[(s.pDice % 6) - 1]:
                s.parent.cPawn[s.pDice % 6 - 1] = score
        else:
            if score < s.parent.cPawn[s.pDice - 1]:
                s.parent.cPawn[s.pDice - 1] = score
    for i in range(len(KL) - 2, 0, -1):
        for s in KL[i]:
            score = getSum(s.cPawn)
            if s.pPawn <= 6:
                if score > s.parent.cPawn[s.pDice % 6 - 1]:
                    s.parent.cPawn[s.pDice % 6 - 1] = score
            else:
                if score < s.parent.cPawn[s.pDice - 1]:
                    s.parent.cPawn[s.pDice - 1] = score
    return KL[0]


def getTheNextStepStatus(SL):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        for i in range(1, 7):
            n, ans = selectPawn(s, i + o)
            for p in ans:
                for m in move:
                    newStatus = tryMakeMove(p, m, s)
                    if newStatus is not False:
                        newStatus.pDice = i
                        NL.append(newStatus)
                    del newStatus
    return NL


def getTheNextStepStatus_updata(SL):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        i = random.randint(1, 6)
        n, ans = selectPawn(s, i + o)
        for p in ans:
            for m in move:
                newStatus = tryMakeMove(p, m, s)
                if newStatus is not False:
                    newStatus.pDice = i
                    NL.append(newStatus)
                del newStatus
    return NL


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


def getTheNextStepStatus_formyvaluenet(SL, p):  # 根据现局面，获得所有合法的后续局面
    NL = []
    if SL[0].pPawn > 6:
        move = ['right', 'down', 'rightdown']
        o = 0
    else:
        move = ['left', 'up', 'leftup']
        o = 6
    for s in SL:
        # i = random.randint(1, 6)
        # n, ans = selectPawn(s, i + o)
        # for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, s)
            if newStatus is not False:
                newStatus.pDice = i
                NL.append(newStatus)
            del newStatus
    return NL


def getSum(L):
    value = 0
    for i in L:
        if i != INFTY and i != -INFTY:
            value += i
    return (1 / 6) * value


def MinimaxGoBack(KL):
    for s in KL[-1]:
        score = getScore(s)
        if s.pPawn > 6:
            if score < s.parent.cPawn[(s.pDice % 6) - 1]:
                s.parent.cPawn[s.pDice % 6 - 1] = score
        else:
            if score > s.parent.cPawn[s.pDice - 1]:
                s.parent.cPawn[s.pDice - 1] = score
    for i in range(len(KL) - 2, 0, -1):
        for s in KL[i]:
            score = getSum(s.cPawn)
            if s.pPawn > 6:
                if score < s.parent.cPawn[s.pDice % 6 - 1]:
                    s.parent.cPawn[s.pDice % 6 - 1] = score
            else:
                if score > s.parent.cPawn[s.pDice - 1]:
                    s.parent.cPawn[s.pDice - 1] = score
    return KL[0]


def BeyesGoBack(KL, rPro=1, bPro=1):
    for s in KL[-1]:
        score = getScore(s)
        if s.pPawn > 6:
            if score < s.parent.cPawn[(s.pDice % 6) - 1]:
                s.parent.cPawnSecond[s.pDice % 6 - 1] = s.parent.cPawn[s.pDice % 6 - 1]
                s.parent.cPawn[s.pDice % 6 - 1] = score
                if len(KL) == 2:
                    s.parent.cPMSecond[s.pDice % 6 - 1] = s.parent.cPM[(s.pDice % 6) - 1]
                    s.parent.cPM[(s.pDice % 6) - 1] = [s.pPawn, s.pMove]
        else:
            if score > s.parent.cPawn[s.pDice - 1]:
                s.parent.cPawn[s.pDice % 6 - 1] = score
                if len(KL) == 2:
                    s.parent.cPM[(s.pDice % 6) - 1] = [s.pPawn, s.pMove]
    for s in KL[-2]:
        if s.pPawn > 6:
            break;
        for d in range(1, 7):
            if abs(s.cPawnSecond[d - 1]) != INFTY:
                if random.random() + bPro * 0.90 < 1:
                    s.cPawn[d - 1] = s.cPawnSecond[d - 1]
                    s.cPM[d - 1] = s.cPMSecond[d - 1]
    for i in range(len(KL) - 2, 0, -1):
        for s in KL[i]:
            score = getSum(s.cPawn)
            if s.pPawn > 6:
                if score < s.parent.cPawn[(s.pDice % 6) - 1]:
                    s.parent.cPawnSecond[s.pDice % 6 - 1] = s.parent.cPawn[s.pDice % 6 - 1]
                    s.parent.cPawn[s.pDice % 6 - 1] = score
                    if i == 1:
                        s.parent.cPMSecond[s.pDice % 6 - 1] = s.parent.cPM[(s.pDice % 6) - 1]
                        s.parent.cPM[(s.pDice % 6) - 1] = [s.pPawn, s.pMove]
            else:
                if score > s.parent.cPawn[s.pDice - 1]:
                    s.parent.cPawnSecond[s.pDice % 6 - 1] = s.parent.cPawn[s.pDice % 6 - 1]
                    s.parent.cPawn[s.pDice % 6 - 1] = score
                    if i == 1:
                        s.parent.cPMSecond[s.pDice % 6 - 1] = s.parent.cPM[(s.pDice % 6) - 1]
                        s.parent.cPM[(s.pDice % 6) - 1] = [s.pPawn, s.pMove]
        for s in KL[i]:
            if s.pPawn > 6:
                break;
            for d in range(1, 7):
                if abs(s.cPawnSecond[d - 1]) != INFTY:
                    if random.random() + bPro * 0.90 < 1:
                        s.cPawn[d - 1] = s.cPMSecond[d - 1]
                        s.cPM[d - 1] = s.cPMSecond[d - 1]
    return KL[0]


def redByBeyes(ans, lastInfo, k=2.2, lam=5, STEP=2):
    global myGuess, matchPro
    if lastInfo != []:
        if check(lastInfo):
            matchPro = updatePro(matchPro, 1)
        else:
            matchPro = updatePro(matchPro, 0)
    # print(matchPro)
    maxValue = theValue = -INFTY
    bestp = 0
    bestm = ''
    move = ['right', 'down', 'rightdown']
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
    STEP -= 1
    if len(SL) == 1:
        bestp, bestm = SL[0].pPawn, SL[0].pMove
    else:
        KL.append(SL)
        for i in range(STEP):
            NL = getTheNextStepStatus(KL[-1])
            KL.append(NL)
        KL = BeyesGoBack(KL, rPro=1, bPro=1)
        for s in KL:
            theValue = getSum(s.cPawn)
            if theValue > maxValue:
                maxValue, bestp, bestm = theValue, s.pPawn, s.pMove
                myGuess = s
    return bestp, bestm


def redByNerualpure(ans, endk=3, value_balance=0.22):
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
    datanew = get_alldata_5(S)
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
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
    out = out.tolist()
    # probli = []
    consider = {}

    for move in SL:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        consider[move] = probli
        k = isEnd(move)
        if k:
            break
    if k:
        bestp = move.pPawn
        bestm = move.pMove
        return bestp, bestm

    "下面考虑加入估值因子"
    k = 0
    valuescore = {}
    for move in SL:
        valuescore[move] = 0
        simigames = 0
        NL = []
        NL.append(move)
        nowpawns = move.pawn
        totalscore = 0

        pro = getPawnProblue(move)
        for i in range(7, 13):
            tempscore = 0
            if nowpawns[i - 1] != 0:
                availables = getTheNextStepStatus_formyvaluenet(NL, i)

                considertemp = {}
                for moved in availables:
                    theValue = getDemoValueblue(moved)
                    considertemp[moved] = theValue

                move_otherside = max(considertemp, key=lambda x: considertemp[x])  # 最大值对应的键
                nowmove = copy.deepcopy(move_otherside)
                del move_otherside

                if nowmove is not False:
                    k = isEnd(nowmove)
                    if k:
                        tempscore = tempscore - 1 * endk
                        simigames = simigames + 1
                if k == 0:
                    datanew = get_alldata_5(nowmove)
                    datanew = np.array(datanew)
                    datanew = torch.tensor(datanew)
                    datanew = datanew.cuda(device)
                    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
                    newscore = valuenet(datanew)[0][0]
                    tempscore = tempscore + newscore
                    simigames = simigames + 1

            tempscore = tempscore * pro[i - 1]
            totalscore = totalscore + tempscore
        if simigames != 0:
            valuescore[move] = totalscore / simigames
        else:
            valuescore[move] = totalscore
        del NL

    for move in SL:
        consider[move] = consider[move] + value_balance * valuescore[move]

    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    print("the policy is ", consider.values())
    print("the value is ", valuescore.values())
    bestp = The_total_choose.pPawn
    bestm = The_total_choose.pMove
    return bestp, bestm


def redByValuePure(ans):
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
    consider = {}
    for move in SL:
        datanew = get_alldata_5(move)
        datanew = np.array(datanew)
        datanew = torch.tensor(datanew)
        datanew = datanew.cuda(device)
        datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
        newscore = valuenet(datanew)[0][0]
        consider[move] = newscore

    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    print("the policy is ", consider.values())
    bestp = The_total_choose.pPawn
    bestm = The_total_choose.pMove
    return bestp, bestm


def redByPolicypure(ans):
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
    datanew = get_alldata_5(S)
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
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
    out = out.tolist()
    # probli = []
    consider = {}

    for move in SL:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 4:move_p * 4 - 1]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        consider[move] = probli
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    print(consider.values())
    bestp = The_total_choose.pPawn
    bestm = The_total_choose.pMove
    return bestp, bestm


def redByNerualpurewithclassify(ans):
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    KL = []
    SL = []
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
    datanew = get_alldata_5(S)
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
    outnew = net(datanew)

    prenew = outnew
    out = torch.zeros(19)
    prenew = prenew.squeeze()
    # out = self.trans(out)
    out = out.cuda(device)
    # print(out)
    for i in range(0, 6):
        oneStatus = prenew[i * 3:(i + 1) * 3]
        out[i * 3:(i + 1) * 3] = 1 * torch.softmax(oneStatus, 0)
    out = out.tolist()
    # probli = []
    consider = {}

    for move in SL:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 3:move_p * 3]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        consider[move] = probli
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    print(consider.values())
    bestp = The_total_choose.pPawn
    bestm = The_total_choose.pMove
    return bestp, bestm


def check(lastInfo):
    n, p, m = lastInfo
    if myGuess.cPM[n % 6 - 1] == []:
        return False
    if myGuess.cPM[n % 6 - 1][0] == p and myGuess.cPM[n % 6 - 1][1] == m:
        return True
    return False


def updatePro(matchPro, flag):
    Pb = matchPro
    Pfb = 1 - matchPro
    Pab = 0.90
    Pfab = 1 - Pab
    Pafb = 0.55
    Pfafb = 1 - Pafb
    if flag:
        Pa = Pab * Pb + Pafb * Pfb
        Pb = (Pab * Pb) / Pa
    else:
        Pfa = Pfab * Pb + Pfafb * Pfb
        Pb = (Pfab * Pb) / Pfa
    Pb = min(0.9645, max(Pb, 0.6555))
    return Pb


def writefile(name_they, remymap, result):
    global mytext, winner
    nowtime1 = time.strftime("%Y%m%d%H%M")
    nowtime2 = time.strftime("%Y.%m.%d %H:%M")
    if playhand == 'first':
        if result == 1:
            winner = '先手胜'
        if result == 2:
            winner = '后手胜'
        mytext = open(
            'WTN-' + 'D5贝塔一' + 'vs' + name_they + '-' + winner + nowtime1 + '.txt',
            'a+')
        mytext.write(
            '#' + '[D5贝塔一]' + '[' + name_they + '][' + winner + ']' + '[' + nowtime2 + '线上' + ']' + '[2023 CCGC];' + '\n')

        mytext.write('R:' + 'A5-6;B5-2;C5-4;A4-1;B4-5;A3-3' + '\n')
        mytext.write(
            'B:' + 'E3-' + str(remymap[0]) + ';D2-' + str(remymap[1]) + ';E2-' + str(remymap[2]) + ';C1-' + str(
                remymap[3]) + ';D1-' + str(remymap[4]) + ';E1-' + str(remymap[5]) + '\n')

        for message in recordplay:
            mytext.write(message + '\n')

    if playhand == 'second':
        if result == 1:
            winner = '后手胜'
        if result == 2:
            winner = '先手胜'
        mytext = open(
            'WTN-' + name_they + 'vs' + 'D5贝塔一' + '-' + winner + nowtime1 + '.txt',
            'a+')
        mytext.write(
            '#' + '[' + name_they + ']' + '[D5贝塔一][' + winner + ']' + '[' + nowtime2 + '线上' + ']' + '[2023 CCGC];' + '\n')

        mytext.write(
            'R:' + 'A5-' + str(remymap[0]) + ';B5-' + str(remymap[1]) + ';C5-' + str(remymap[2]) + ';A4-' + str(
                remymap[3]) + ';B4-' + str(remymap[4]) + ';A3-' + str(remymap[5]) + '\n')
        mytext.write('B:' + 'E3-3;D2-5;E2-1;C1-4;D1-2;E1-6' + '\n')

        for message in recordplay:
            mytext.write(message + '\n')
    mytext.close()


def startGame(Red, Blue, n, detail):
    global COUNT
    global S
    global playsr
    global winsr
    global playhand
    global remymap
    global recordplay
    global recordtime

    playhand = 'None'
    remymap = []
    name_they = input('请输入对手的名字：')
    init()
    RESULT[0] = 0
    RESULT[1] = 0
    cnt = n
    rateline = []

    while cnt:
        recordplay = []
        recordtime = 0
        c = input('请输入先行棋的一方，红1蓝2: ')
        c = int(c)
        COUNT = c
        if c == 1:
            playhand = 'first'
        else:
            playhand = 'second'

        global S
        S = Status()
        resetInfo()
        playsr = {}
        winsr = {}
        result = playGame(Red, Blue, detail, now=allcnt - cnt)  # 游戏开始，返回比赛结果
        gc.collect()

        writefile(name_they, remymap, result)
        print("Game is in " + str(allcnt - cnt))
        print("The result is ", result)
        del S
        del winsr
        del playsr
        if detail:
            pass
        RESULT[result - 1] += 1  # 更新比分
        cnt -= 1
        COUNT = 2000 - cnt - 1  # 先手方交替
        rateline.append(float(RESULT[0]) / sum(RESULT))
        if cnt % 7 == 0:  # 每五轮显示一次胜负情况
            print(sum(RESULT), '\t', round(100 * RESULT[0] / sum(RESULT), 4))
    return round(100 * RESULT[0] / sum(RESULT), 4)


# def startGame2(Red, Blue, n, filename, detail):
#     global COUNT
#     global S
#     global playsr
#     global winsr
#     global playhand
#     global remymap
#     global recordplay
#     global recordtime
#     playhand = 'None'
#     remymap = []
#
#     name_they = input('输入对方的名字: ')
#     init()
#     RESULT[0] = 0
#     RESULT[1] = 0
#     cnt = n
#     rateline = []
#     while cnt:
#         recordplay = []
#         recordtime = 0
#         c = input('请输入先行棋的一方，红1蓝2: ')
#         c = int(c)
#         COUNT = c
#         if c == 1:
#             playhand = 'first'
#         else:
#             playhand = 'second'
#         global S
#         S = Status()
#         resetInfo()
#         playsr = {}
#         winsr = {}
#
#         result = playGame(Red, Blue, detail, now=allcnt - cnt)  # 游戏开始，返回比赛结果
#         gc.collect()
#
#         writefile(name_they=name_they, remymap=remymap, result=result)
#
#         # print(globals())
#         we_result.append(result)
#         # np.save('results/result0714', we_result)
#         # np.save('0705_result', we_result)
#         print("game is in" + str(allcnt - cnt))
#         print("the result is ", result)
#         del S
#         del winsr
#         del playsr
#         if detail:
#             pass
#         RESULT[result - 1] += 1  # 更新比分
#         cnt -= 1
#         COUNT = 2000 - cnt - 1  # 先手方交替
#         rateline.append(float(RESULT[0]) / sum(RESULT))
#         if cnt % 5 == 0:  # 每五轮显示一次胜负情况
#             print(sum(RESULT), '\t', round(100 * RESULT[0] / sum(RESULT), 4))
#     return RESULT[0]


def getalldata():  # 红是1-6，蓝是7-12
    datanew = []
    NPmap = np.array(S.map)
    reddata = np.where(NPmap < 7, NPmap, 0)
    bluedata = np.where(NPmap >= 7, NPmap, 0)
    datanew.append(reddata)
    datanew.append(bluedata)
    label = []
    for i in range(0, 6):
        newnewS = copy.deepcopy(S)
        # global newnewS
        if newnewS.pawn[i] == 0:
            label_data = np.zeros((5, 5), float)
            label.append(label_data)
            del newnewS
        else:
            label_data = NewRedByUct(newnewS.pawn[i], newnewS)
            label.append(label_data)
            del newnewS
    return datanew, label


# 下面是自添加算法
def tryMakeMovenew(p, PawnMoveTo, S):  # 尝试移动，并且返回移动后的棋局地图与棋子存活情况
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
    if notInMap(row + x, col + y):
        return False
    newS.map[row][col] = 0
    row = row + x
    col = col + y
    if newS.map[row][col] != 0:  # 检查移动的目标格子位是否有棋子，是的话被吃掉，赋值为0
        i = newS.pawn.index(newS.map[row][col])
        newS.pawn[i] = 0
    newS.map[row][col] = p
    value = getLocValue(newS)  # 获取所有棋子的位置价值
    newS.pro = getPawnPro(newS)  # 获取所有棋子被摇到的概率
    newS.value = getPawnValue(value, newS.pro)
    newS.parent = S
    newS.pPawn = p
    newS.pMove = PawnMoveTo
    if p < 7:
        newS.cPawn = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
        newS.cPawnSecond = [INFTY, INFTY, INFTY, INFTY, INFTY, INFTY]
    else:
        newS.cPawn = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
        newS.cPawnSecond = [-INFTY, -INFTY, -INFTY, -INFTY, -INFTY, -INFTY]
    return newS


def NewRedByUct(ans, newnewS):
    label_data = np.zeros((5, 5), float)
    global newi
    calculation_time = float(15)
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    playsr = {}
    winsr = {}
    SL = []  # 当前局面下合法的后续走法
    p = ans
    for m in move:
        newStatus = tryMakeMovenew(p, m, newnewS)
        if newStatus is not False:
            SL.append(newStatus)
    games = 0
    begin = time.time()
    while time.time() - begin < calculation_time * 5:
        winsr, playsr, availables = run_simulation2(SL, playsr, winsr)
        games += 1
    for move in SL:
        pasPawn = move.pPawn
        NDmap = np.array(move.map)
        posx, posy = np.where(NDmap == pasPawn)
        label_data[posx[0]][posy[0]] = (winsr.get(move, 0) / playsr.get(move, 1))
    return label_data


# 下面是uct算法

def redByUct(ans):
    # 制造红蓝棋谱
    datanew = []
    NPmap = np.array(S.map)
    reddata = np.where(NPmap < 7, NPmap, 0)
    bluedata = np.where(NPmap >= 7, NPmap, 0)
    datanew.append(reddata)
    datanew.append(bluedata)

    calculation_time = float(15)
    bestp = 0;
    bestm = '';
    move = ['right', 'down', 'rightdown']
    playsr = {}
    winsr = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
            del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    while time.time() - begin < calculation_time:
        winsr, playsr, availables = run_simulation2(SL, playsr, winsr)
        games += 1
    bestp, bestm = select_one_move(winsr, playsr, SL)
    for move1 in SL:
        print(winsr.get(move1, 0) / playsr.get(move1, 1))
    print(games)
    print(bestp, bestm)
    return bestp, bestm


def redByUctPlusDemo(ans):
    global playsr
    global winsr
    global Vsr
    global recorder
    global Vsr_all
    print("red can move ", ans)
    calculation_time = float(10)
    bestp = 0
    bestm = ''
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    move = ['right', 'down', 'rightdown']
    Vsr = {}
    Vsr_all = {}
    recorder = {}
    Tempt = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
                del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    playsr[S] = 0

    while time.time() - begin < calculation_time:  # time.time() - begin < calculation_time
        playsr, winsr = run_simulationWithDemo(SL, games, 1, 1000)
        games += 1
        if games % 120 == 0:
            for move in SL:
                Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)
                # file.write(str(Tempt[move]) + ' ')
            print(Tempt)
            # file.write('\n')
    playsr[S] = 0
    # file.close()
    for move in SL:
        Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)
    # for move in SL:
    #     Tempt[move] = playsr.get(move,0)
    move_choose = max(Tempt, key=lambda x: Tempt[x])

    for move in SL:
        if move != move_choose and Tempt[move] == Tempt[move_choose] and move.pMove == 'rightdown':
            move_choose = move
    bestp = move_choose.pPawn
    bestm = move_choose.pMove
    # bestp, bestm = select_one_move_nerual(SL)
    for move in SL:
        print(playsr.get(move, 0))
    for move1 in SL:
        print(Tempt.get(move1, 0))
    # for item in consider.values():
    #     print("The total C are", item)
    # for move in SL:
    #     print("The output of net are", for_print.get(move, 0))
    print('we have searched for ', games)
    print(bestp, bestm)

    del Vsr
    del Vsr_all
    del recorder
    del SL
    gc.collect()
    return bestp, bestm


def winner(ans):
    global playsr
    global winsr
    global Vsr
    global recorder
    global Vsr_all
    print("red can move ", ans)
    calculation_time = float(60)
    bestp = 0
    bestm = ''
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    move = ['right', 'down', 'rightdown']
    Vsr = {}
    Vsr_all = {}
    recorder = {}
    Tempt = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
                del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    playsr[S] = 0

    while time.time() - begin < calculation_time:  # time.time() - begin < calculation_time
        playsr, winsr = run_simulationwinner(SL, games, 1, 1000)
        games += 1
        # for move1 in SL:
        #     print("These followings are the process of searching")
        #     print(winsr.get(move1, 0) / playsr.get(move1, 1))
    playsr[S] = 0
    for move in SL:
        Tempt[move] = winsr.get(move, 0) / playsr.get(move, 1)
    # for move in SL:
    #     Tempt[move] = playsr.get(move,0)
    move_choose = max(Tempt, key=lambda x: Tempt[x])

    for move in SL:
        if move != move_choose and Tempt[move] == Tempt[move_choose] and move.pMove == 'rightdown':
            move_choose = move
    bestp = move_choose.pPawn
    bestm = move_choose.pMove
    # bestp, bestm = select_one_move_nerual(SL)
    for move in SL:
        print(playsr.get(move, 0))
    for move1 in SL:
        print(Tempt.get(move1, 0))
    # for item in consider.values():
    #     print("The total C are", item)
    # for move in SL:
    #     print("The output of net are", for_print.get(move, 0))
    print('we have searched for ', games)
    print(bestp, bestm)

    del Vsr
    del Vsr_all
    del recorder
    del SL
    gc.collect()
    return bestp, bestm


def blueByUct(ans):
    print("blue can move ", ans)
    calculation_time = float(120)
    bestp = 0;
    bestm = '';
    move = ['left', 'up', 'leftup']
    plays = {}
    wins = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S)
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
            del newStatus
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()

    while time.time() - begin < calculation_time:
        wins, plays, availables = run_simulation(SL, plays, wins)
        games += 1
    for move in SL:
        print(wins.get(move, 0) / plays.get(move, 1))
    print('we have searched for ', games)
    bestp, bestm = select_one_move(wins, plays, SL)
    del SL
    return bestp, bestm


def run_simulation(SL, plays, wins, max_actions=1000):  # 蓝方
    expand = True
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    for t in range(1, max_actions + 1):
        if all(plays.get(move) for move in availables):  # 如果都访问过
            a = 0
            for move in availables:
                a += plays.get(move)  # 总访问次数
            move = availables[0]
            for moved in availables:
                if ((wins.get(move) / (plays.get(move) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(move) + 1e-99)) ** 0.5) < (
                        (wins.get(moved) / (plays.get(moved) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(moved) + 1e-99)) ** 0.5):
                    move = moved

        else:  # 随机选一个拓展
            peripherals = []
            for move in availables:
                if not plays.get(move):
                    peripherals.append(move)
            move = choice(peripherals)
        # bestp,bestm=move.pPawn,move.pMove
        NL = []
        NL.append(move)
        availables = getTheNextStepStatus(NL)  # 更新合法后继
        if expand and move not in plays:
            expand = False
            plays[move] = 0
            wins[move] = 0
        visited_states.add(move)
        k = 0
        if move is not False:
            k = isEnd(move)
            if k:
                break
    for move in visited_states:
        if move in plays:
            plays[move] += 1
            # all visited moves
            if k == 2:
                wins[move] += 1
    return wins, plays, availables


def run_simulation2(SL, plays, wins, max_actions=1000):  # 红方
    expand = True
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    for t in range(1, max_actions + 1):
        if all(plays.get(move) for move in availables):  # 如果都访问过
            a = 0
            for move in availables:
                a += plays.get(move)  # 总访问次数
            move = availables[0]
            for moved in availables:
                if ((wins.get(move) / (plays.get(move) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(move) + 1e-99)) ** 0.5 + 1.15 * getScore(move) / (
                            plays.get(move))) < ((wins.get(moved) / (plays.get(moved) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (plays.get(moved) + 1e-99)) ** 0.5 + 1.15 * getScore(moved) / plays.get(
                    moved)):
                    move = moved

        else:  # 随机选一个拓展
            peripherals = []
            for move in availables:
                if not plays.get(move):
                    peripherals.append(move)
            move = choice(peripherals)
        NL = []
        NL.append(move)
        availables = getTheNextStepStatus(NL)  # 更新合法后继
        if expand and move not in plays:
            expand = False
            plays[move] = 0
            wins[move] = 0
        visited_states.add(move)
        k = 0
        if move is not False:
            k = isEnd(move)
            if k:
                break
    for move in visited_states:
        if move in plays:
            plays[move] += 1
            # all visited moves
            if k == 1:
                wins[move] += 1
    return wins, plays, availables


# 下面是rave和z检验
def blueByUctrave(ans):
    calculation_time = float(15)
    bestp = 0;
    bestm = '';
    move = ['left', 'up', 'leftup']
    plays = {}
    wins = {}
    plays_rave = {}  # key:move, value:visited times
    wins_rave = {}
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):
                    return p, m
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    games = 0
    begin = time.time()
    while time.time() - begin < calculation_time:
        wins, plays, plays_rave, wins_rave, availables = run_simulationrave(SL, plays, wins, plays_rave, wins_rave)
        games += 1
    bestp, bestm = select_one_move(wins, plays, SL)
    return bestp, bestm


def run_simulationrave(SL, plays, wins, plays_rave, wins_rave, max_actions=1000):  # 蓝方
    expand = True
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    for t in range(1, max_actions + 1):
        if all(plays.get(move) for move in availables):  # 如果都访问过
            a = 0
            for move in availables:
                a += plays.get(move)  # 总访问次数
            move = availables[0]
            for moved in availables:
                if ((1 - sqrt(a / (3 * plays_rave.get(move) + a))) * (wins.get(move) / plays.get(move)) +
                    sqrt(a / (3 * plays_rave.get(move) + a)) * (wins_rave.get(move) / plays_rave.get(move)) +
                    sqrt(1.96 * log(plays_rave.get(move)) / plays.get(move))) < (
                        (1 - sqrt(a / (3 * plays_rave.get(moved) + a))) * (wins.get(moved) / plays.get(moved)) +
                        sqrt(a / (3 * plays_rave.get(moved) + a)) * (
                                wins_rave.get(moved) / plays_rave.get(moved)) + sqrt(
                    1.96 * log(plays_rave.get(moved)) / plays.get(moved))):
                    move = moved
        else:  # 随机选一个拓展
            peripherals = []
            for move in availables:
                if not plays.get(move):
                    peripherals.append(move)
            move = choice(peripherals)
        # bestp,bestm=move.pPawn,move.pMove
        NL = []
        NL.append(move)
        availables = getTheNextStepStatus(NL)  # 更新合法后继
        if expand and move not in plays:
            expand = False
            plays[move] = 0
            wins[move] = 0
            if move not in plays_rave:
                plays_rave[move] = 0
            if move in wins_rave:
                wins_rave[move] = 0
            else:
                wins_rave[move] = 0
        visited_states.add(move)
        k = 0
        if move is not False:
            k = isEnd(move)
            if k:
                break
    for move in visited_states:
        if move in plays:
            plays[move] += 1
            if k == 2:
                wins[move] += 1
        if move in plays_rave:
            plays_rave[move] += 1  # no matter which player
            if move in wins_rave[move]:
                wins_rave[move] += 1
    return wins, plays, availables


def blueByUct_z_prune(ans):
    calculation_time = float(15)
    bestp = 0;
    bestm = '';
    plays = {}
    wins = {}
    b = time.time()
    wins, plays, SL = z_prune(wins, plays, ans)  # 剪枝
    if len(SL) == 1:
        return SL[0].pPawn, SL[0].pMove  # 如只有一个合法后续，直接返回
    begin = time.time()
    while time.time() - begin < calculation_time:
        wins, plays, availables = run_simulation(SL, plays, wins)
    bestp, bestm = select_one_move(wins, plays, SL)
    for move1 in SL:
        print(wins.get(move1, 0) / plays.get(move1, 1))
    print(bestp, bestm)
    return bestp, bestm


def z_prune(wins, plays, ans):  # 返回的是SL
    cnum = 30  # 模拟次数
    move = ['left', 'up', 'leftup']
    SL = []  # 当前局面下合法的后续走法
    for p in ans:
        for m in move:
            newStatus = tryMakeMove(p, m, S);
            if newStatus is not False:
                SL.append(newStatus)
                if isEnd(newStatus):  # 如果获胜直接返回
                    LS = []
                    LS.append(SL[-1])
                    return wins, plays, LS
    if len(SL) == 1:
        return wins, plays, SL
    else:  # 开始剪枝
        values1 = []  # 估值平均数
        standards = []  # 标准差
        LS = SL[::]
        print('最开始的LS:', LS)
        for move in SL:
            nnum = 0
            score_all = []  # 每个节点估值的集合
            while nnum < cnum:  # 每个节点按次数限制
                wins, plays, scores = run_simulation1(move, plays, wins)
                nnum += 1
                score_all.append(scores)
            score_all = np.array(score_all)
            x_min = np.sum(score_all) / cnum  # 该节点
            standard = np.std(score_all, ddof=1) ** 2
            values1.append(x_min)
            standards.append(standard)
        values1 = np.array(values1)
        standards = np.array(standards)
        a = np.argmax(values1)
        for i in range(0, 3):
            if i != a and values1[i] != False:
                z = (values1[a] - values1[i] - 8) / math.sqrt((standards[a] + standards[i]) / cnum)
                print('z=', z)
                if z >= 1.645:
                    LS.remove(SL[i])
        return wins, plays, LS


def run_simulation1(move, plays, wins, max_actions=10):  # 预模拟，返回评估值
    newplays = plays
    newwins = wins
    expand = True
    score1 = 0
    availables = []  # 合法后继
    availables.append(move)
    visited_states = set()  # 以访问节点，判断是否拓展
    for t in range(1, max_actions + 1):
        if all(newplays.get(move) for move in availables):
            a = 0
            for move in availables:
                a += newplays.get(move)  # 总访问次数
            move = availables[0]
            for moved in availables:
                if ((newwins.get(move) / (newplays.get(move) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (newplays.get(move) + 1e-99)) ** 0.5 + 1.15 * getDemoValue(move) / (
                            newplays.get(move))) < ((newwins.get(moved) / (newplays.get(moved) + 1e-99)) + 0.85 * (
                        2 * math.log(a) / (newplays.get(moved) + 1e-99)) ** 0.5 + 1.15 * getDemoValue(
                    moved) / newplays.get(moved)):
                    move = moved  # 此时move为选择
        else:
            peripherals = []
            for move in availables:
                if not newplays.get(move):
                    peripherals.append(move)
            move = choice(peripherals)  # 此时move为选择
        NL = []
        NL.append(move)
        availables = getTheNextStepStatus(NL)
        if expand and move not in plays:
            expand = False
            newplays[move] = 0
            newwins[move] = 0
        visited_states.add(move)
        k = 0
        if move is not False:
            k = isEnd(move)
            if k:
                break
    for move1 in visited_states:
        if move1 in newplays:
            newplays[move1] += 1
            # all visited moves
            if k == 2:
                newwins[move1] += 1
    score1 = getDemoValue(move, 1)
    return newwins, newplays, score1


def select_one_move(wins, plays, availables):
    move = availables[0]
    for moved in availables:
        if wins.get(move, 0) <= wins.get(moved, 0):
            move = moved
    return move.pPawn, move.pMove


def get_standard_data(newS: Status):
    newdata = np.array(newS.map)
    reddata = np.where((newdata < 13) & (newdata > 6), newdata, 0)
    bluedata = np.where((newdata < 7) & (newdata > 0), newdata, 0)
    return reddata, bluedata


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


def redByNeuralUCT(ans):
    global playsr
    global winsr
    # global Vsr
    # global recorder
    # global Vsr_all
    # print("red can move ", ans)
    calculation_time = float(10)
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
    num_process = 1  # 多进程数目
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


def cal_choice(probli, nowS, c=1, lamda=0.5):
    global playsr
    global winsr
    U = c * probli * (playsr.get(nowS.parent, 0) + 1) ** lamda / (1 + playsr.get(nowS, 0))
    W = winsr.get(nowS, 0) / playsr.get(nowS, 1)
    C = U + W
    return C


def cal_first_choice(probli, Vsr, nowS, c=9, value_balance=0.1, _playsr=None):
    if _playsr is None:
        _playsr = {}
    U = c * probli / (_playsr.get(nowS, 0) + 1)
    V = Vsr.get(nowS, 0) * value_balance
    C = U + V
    return C


def select_one_move_nerual(availables):
    move = availables[0]
    for moved in availables:
        if winsr.get(move, 0) / playsr.get(move, 100000) <= winsr.get(moved, 0) / playsr.get(move, 100000):
            move = moved
    return move.pPawn, move.pMove


def reverse(matrix):
    b = np.reshape(matrix, -1)
    c = b[::-1]
    d = np.reshape(c, [5, 5])
    return d


def get_alldata_5blue(nowS):
    datanew = []
    NPmap = reverse(np.array(nowS.map))
    reddata_toblue = np.where((NPmap < 7) & (NPmap > 0), NPmap + 6, np.zeros([5, 5]))
    bluedata_tored = np.where((NPmap >= 7), (NPmap - 6), np.zeros([5, 5]))
    datanew.append(bluedata_tored)
    datanew.append(reddata_toblue)

    if nowS.parent is not None:
        NPmap = reverse(np.array(nowS.parent.map))
        reddata_toblue = np.where(NPmap < 7, NPmap + 6, 0)
        bluedata_tored = np.where(NPmap >= 7, NPmap - 6, 0)
        datanew.append(bluedata_tored)
        datanew.append(reddata_toblue)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if nowS.parent_before is not None:
        NPmap = reverse(np.array(nowS.parent_before))
        reddata_toblue = np.where(NPmap < 7, NPmap + 6, 0)
        bluedata_tored = np.where(NPmap >= 7, NPmap - 6, 0)
        datanew.append(bluedata_tored)
        datanew.append(reddata_toblue)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if nowS.parent_3 is not None:
        NPmap = reverse(np.array(nowS.parent_3))
        reddata_toblue = np.where(NPmap < 7, NPmap + 6, 0)
        bluedata_tored = np.where(NPmap >= 7, NPmap - 6, 0)
        datanew.append(bluedata_tored)
        datanew.append(reddata_toblue)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)

    if nowS.parent_4 is not None:
        NPmap = reverse(np.array(nowS.parent_4))
        reddata_toblue = np.where(NPmap < 7, NPmap + 6, 0)
        bluedata_tored = np.where(NPmap >= 7, NPmap - 6, 0)
        datanew.append(bluedata_tored)
        datanew.append(reddata_toblue)
    else:
        reddata = np.zeros([5, 5], int)
        bluedata = np.zeros([5, 5], int)
        datanew.append(reddata)
        datanew.append(bluedata)
    return datanew


def run_simulationWithDemo(SL, games, c=1, max_actions=1000):  # 红方
    global winsr
    global playsr
    global S
    global Vsr
    global Vsr_all
    global recorder
    movetored = ['right', 'down', 'rightdown']
    movetoblue = ['left', 'up', 'leftup']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    moveblue_movered = {'left': 'right', 'up': 'down', 'leftup': 'rightdown'}
    Qsr = {}
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    visited_red = set()
    playsr[S] += 1

    move = availables[0]
    for moved in availables:
        if ((winsr.get(move, 0) / (playsr.get(move, 1))) + 0.85 * (
                2 * playsr.get(S, 0) / (playsr.get(move, 1))) ** 0.5) < (
                (winsr.get(moved, 0) / (playsr.get(moved, 1))) + 0.85 * (
                2 * playsr.get(S, 0) / (playsr.get(moved, 1))) ** 0.5):
            move = moved

    The_total_choose = move

    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    if playsr.get(The_total_choose, -1) == -1:
        playsr[The_total_choose] = 0
        winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):

                NL = []
                NL.append(move_choose)
                availables = getTheNextStepStatus_updata(NL)
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

                # max_problis = torch.tensor(max_problis)
                # print('torch_softmax',torch.softmax(max_problis/temperature,0))
                max_problis = np.array(max_problis)
                # print('SoftMax',SoftMax(max_problis / temperature))
                move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature), k=1)
                move_otherside = move_otherside[0]
                # print(np.array(move_choose.map))
                # move_otherside = choice(availables)  # 对面的走法运用模拟出来的UCT
                visited_states.add(move_otherside)
                if playsr.get(move_otherside, -1) == -1:
                    playsr[move_otherside] = 0
                    winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                del max_consider
                NL = []
                NL.append(move_otherside)
                availables = getTheNextStepStatus_updata(NL)
                del NL

                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            move_choose = move
                            break
                if k:
                    break
                consider = {}
                max_problis = []
                max_consider = []
                for move in availables:
                    theValue = getDemoValuered(move)
                    consider[move] = theValue
                    max_problis.append(theValue)
                    max_consider.append(move)
                # max_problis = torch.tensor(max_problis)
                max_problis = np.array(max_problis)
                move_choose = random.choices(max_consider, SoftMax(max_problis / temperature), k=1)
                move_choose = move_choose[0]
                visited_red.add(move_choose)
                # print(np.array(move_choose.map))
                # 激活已选节点在playsr中，winsr中
                if playsr.get(move_choose, -1) == -1:
                    playsr[move_choose] = 0
                    winsr[move_choose] = 0
                visited_states.add(move_choose)
                del max_consider
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                # print(np.array(move_choose.map))
                # NL = []
                # NL.append(move_choose)
                # availables = getTheNextStepStatus(NL)  # 更新合法后继
                # del NL
                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    for move in visited_states:
        playsr[move] += 1
        if k == 1:
            winsr[move] += 1

    del visited_states
    del visited_red
    return playsr, winsr


def run_simulationwinner(SL, games, c=1, max_actions=1000):  # 红方
    global winsr
    global playsr
    global S
    global Vsr
    global Vsr_all
    global recorder
    movetored = ['right', 'down', 'rightdown']
    movetoblue = ['left', 'up', 'leftup']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    moveblue_movered = {'left': 'right', 'up': 'down', 'leftup': 'rightdown'}
    Qsr = {}
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    visited_red = set()
    playsr[S] += 1

    move = availables[0]
    for moved in availables:
        if ((winsr.get(move, 0) / (playsr.get(move, 1))) + 0.85 * (
                2 * playsr.get(S, 0) / (playsr.get(move, 1))) ** 0.5) < (
                (winsr.get(moved, 0) / (playsr.get(moved, 1))) + 0.85 * (
                2 * playsr.get(S, 0) / (playsr.get(moved, 1))) ** 0.5):
            move = moved

    The_total_choose = move

    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    if playsr.get(The_total_choose, -1) == -1:
        playsr[The_total_choose] = 0
        winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):

                NL = []
                NL.append(move_choose)
                availables = getTheNextStepStatus_updata(NL)
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
                move_otherside = moved = availables[0]
                ORValue = getDemoValueblue(moved)
                for move in availables:
                    theValue = getDemoValueblue(move)
                    if theValue > ORValue:
                        move_otherside = move

                # max_problis = torch.tensor(max_problis)
                # print('torch_softmax',torch.softmax(max_problis/temperature,0))
                # max_problis = np.array(max_problis)
                # print('SoftMax',SoftMax(max_problis / temperature))
                # move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature), k=1)
                # move_otherside = move_otherside[0]
                # print(np.array(move_choose.map))
                # move_otherside = choice(availables)  # 对面的走法运用模拟出来的UCT
                visited_states.add(move_otherside)
                if playsr.get(move_otherside, -1) == -1:
                    playsr[move_otherside] = 0
                    winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                del max_consider
                NL = []
                NL.append(move_otherside)
                availables = getTheNextStepStatus_updata(NL)
                del NL

                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            move_choose = move
                            break
                if k:
                    break
                consider = {}
                max_problis = []
                max_consider = []
                move_choose = moved = availables[0]
                ORValue = getDemoValuered(moved)

                for move in availables:
                    theValue = getDemoValuered(move)
                    if theValue > ORValue:
                        move_choose = move
                # max_problis = torch.tensor(max_problis)
                # max_problis = np.array(max_problis)
                # move_choose = random.choices(max_consider, SoftMax(max_problis / temperature), k=1)
                # move_choose = move_choose[0]
                visited_red.add(move_choose)
                # print(np.array(move_choose.map))
                # 激活已选节点在playsr中，winsr中
                if playsr.get(move_choose, -1) == -1:
                    playsr[move_choose] = 0
                    winsr[move_choose] = 0
                visited_states.add(move_choose)
                del max_consider
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                # print(np.array(move_choose.map))
                # NL = []
                # NL.append(move_choose)
                # availables = getTheNextStepStatus(NL)  # 更新合法后继
                # del NL
                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    for move in visited_states:
        playsr[move] += 1
        if k == 1:
            winsr[move] += 1

    del visited_states
    del visited_red
    return playsr, winsr

def run_model(model, datanew):
    s = time.time()
    out = model(datanew)
    cost_time = round((time.time() - s) * 1000, 5)  # 毫秒级精度
    return cost_time

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
    net = ResNet(BasicBlock, [1, 1, 1, 1])
    net = net.cuda(device)
    net.load_state_dict(torch.load('0830_1111_Alpha1.pt'))
    net.eval()

    # net_jit = torch.jit.load("Alpha1.pt")

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


def run_simulation_value(SL, datanew, games, c=1, max_actions=1000):  # 红方
    global winsr
    global playsr
    global S
    global Vsr
    global Vsr_all
    global recorder
    movetored = ['right', 'down', 'rightdown']
    movetoblue = ['left', 'up', 'leftup']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    moveblue_movered = {'left': 'right', 'up': 'down', 'leftup': 'rightdown'}
    Qsr = {}
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    visited_red = set()
    playsr[S] += 1
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
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
        if games == 0:
            print(move.pMove, probli)
        C = cal_first_choice(probli, Vsr, move, 9)
        consider[move] = C
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    del consider
    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    if playsr.get(The_total_choose, -1) == -1:
        playsr[The_total_choose] = 0
        winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):

                NL = []
                NL.append(move_choose)
                availables = getTheNextStepStatus_updata(NL)
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

                # max_problis = torch.tensor(max_problis)
                # print('torch_softmax',torch.softmax(max_problis/temperature,0))
                max_problis = np.array(max_problis)
                # print('SoftMax',SoftMax(max_problis / temperature))
                move_otherside = random.choices(max_consider, SoftMax(max_problis / temperature), k=1)
                move_otherside = move_otherside[0]
                # print(np.array(move_choose.map))
                # move_otherside = choice(availables)  # 对面的走法运用模拟出来的UCT
                visited_states.add(move_otherside)
                if playsr.get(move_otherside, -1) == -1:
                    playsr[move_otherside] = 0
                    winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                del max_consider
                NL = []
                NL.append(move_otherside)
                availables = getTheNextStepStatus_updata(NL)
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
                # move_choose = choice(max_consider)
                # Qsr[move_choose] = valuenet(datanew)[0][0]
                Qsr[move_choose] = 0
                visited_red.add(move_choose)

                # 激活已选节点在playsr中，winsr中
                if playsr.get(move_choose, -1) == -1:
                    playsr[move_choose] = 0
                    winsr[move_choose] = 0
                visited_states.add(move_choose)
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                # print(np.array(move_choose.map))
                # NL = []
                # NL.append(move_choose)
                # availables = getTheNextStepStatus(NL)  # 更新合法后继
                # del NL
                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    for move in visited_states:
        playsr[move] += 1
        if k == 1:
            winsr[move] += 1
        if move in visited_red:
            if recorder.get(The_total_choose, 0) == 0:
                recorder[The_total_choose] = 0
            recorder[The_total_choose] += 1
            if k == 1:
                Qsr[move] = (Qsr.get(move, 0) + 1) / 2
            if k == 2:
                Qsr[move] = (Qsr.get(move, 0) - 1) / 2
            Vsr_all[The_total_choose] += Qsr.get(move, 0)
        Vsr[The_total_choose] = Vsr_all.get(The_total_choose, 0) / recorder.get(The_total_choose, 1)
    del visited_states
    del visited_red
    return playsr, winsr


def run_simulation_network_classify(SL, datanew, games, c=1, max_actions=1000):  # 红方
    global winsr
    global playsr
    global S
    global Vsr
    global Vsr_all
    global recorder
    movetored = ['right', 'down', 'rightdown']
    movetoblue = ['left', 'up', 'leftup']
    move_dict = {'right': 0, 'down': 1, 'rightdown': 2}
    moveblue_movered = {'left': 'right', 'up': 'down', 'leftup': 'rightdown'}
    Qsr = {}
    availables = SL  # 合法后继
    visited_states = set()  # 以访问节点，判断是否拓展
    visited_red = set()
    playsr[S] += 1
    datanew = np.array(datanew)
    datanew = torch.tensor(datanew)
    datanew = datanew.cuda(device)
    datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
    outnew = net(datanew)

    prenew = outnew
    out = torch.zeros(19)
    prenew = prenew.squeeze()
    # out = self.trans(out)
    out = out.cuda(device)
    # print(out)
    for i in range(0, 6):
        oneStatus = prenew[i * 3:(i + 1) * 3]
        out[i * 3:(i + 1) * 3] = 1 * torch.softmax(oneStatus, 0)
        for index, value in enumerate(out[i * 3:(i + 1) * 3]):
            if index < 3:
                if value < 0.1:
                    out[i * 3 + index] = 0.1
    out = out.tolist()
    # probli = []
    consider = {}

    for move in availables:
        move_p = move.pPawn
        move_to = move.pMove
        problis = out[(move_p - 1) * 3:move_p * 3]  # move = ['right', 'down', 'rightdown']
        probli = problis[move_dict[move_to]]
        if games == 0:
            print(move.pMove, probli)
        C = cal_first_choice(probli, Vsr, move, 9)
        consider[move] = C
    The_total_choose = max(consider, key=lambda x: consider[x])  # 最大值对应的键
    del consider
    move_choose = copy.deepcopy(The_total_choose)

    # 激活已选节点在playsr中，winsr中
    if playsr.get(The_total_choose, -1) == -1:
        playsr[The_total_choose] = 0
        winsr[The_total_choose] = 0
    visited_states.add(The_total_choose)
    k = 0
    if move_choose is not False:
        k = isEnd(move_choose)
        if k == False:
            for t in range(1, max_actions):

                NL = []
                NL.append(move_choose)
                availables = getTheNextStepStatus_updata(NL)
                del NL
                for move in availables:
                    if move is not False:
                        k = isEnd(move)
                        if k:
                            move_otherside = move
                            break
                if k:
                    break
                datanew = get_alldata_5blue(move_choose)

                datanew = np.array(datanew)
                datanew = torch.tensor(datanew)
                datanew = datanew.cuda(device)
                datanew = Variable(torch.unsqueeze(datanew, dim=0).float(), requires_grad=False)
                outnew = net(datanew)

                prenew = outnew
                out = torch.zeros(19)
                prenew = prenew.squeeze()
                # out = self.trans(out)
                out = out.cuda(device)
                # print(out)
                for i in range(0, 6):
                    oneStatus = prenew[i * 3:(i + 1) * 3]
                    out[i * 3:(i + 1) * 3] = 1 * torch.softmax(oneStatus, 0)
                    for index, value in enumerate(out[i * 3:(i + 1) * 3]):
                        if index < 3:
                            if value < 0.1:
                                out[i * 3 + index] = 0.1
                out = out.tolist()
                # probli = []
                consider = {}
                max_problis = []
                max_consider = []
                for move in availables:
                    move_p = move.pPawn - 6
                    move_to = moveblue_movered[move.pMove]
                    problis = out[(move_p - 1) * 3:move_p * 3]  # move = ['right', 'down', 'rightdown']
                    # max_probli = max(problis)
                    probli = problis[move_dict[move_to]]
                    max_problis.append(probli)
                    C = cal_choice(probli, move, 1, 2)

                    consider[move] = probli
                list_consider_value = list(consider.values())
                list_consider_keys = list(consider.keys())
                for one_pro in max_problis:
                    position = list_consider_value.index(one_pro)
                    max_consider.append(list_consider_keys[position])

                max_problis = torch.tensor(max_problis)
                move_otherside = random.choices(max_consider, softmax(max_problis / temperature, 0), k=1)
                move_otherside = move_otherside[0]
                # move_otherside = choice(availables)  # 对面的走法运用模拟出来的UCT
                visited_states.add(move_otherside)
                del max_consider
                del list_consider_keys
                del list_consider_value
                if playsr.get(move_otherside, -1) == -1:
                    playsr[move_otherside] = 0
                    winsr[move_otherside] = 0
                if move_otherside is not False:
                    k = isEnd(move_otherside)
                    if k:
                        break
                NL = []
                NL.append(move_otherside)
                availables = getTheNextStepStatus_updata(NL)
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

                prenew = outnew
                out = torch.zeros(25)
                prenew = prenew.squeeze()
                # out = self.trans(out)
                out = out.cuda(device)
                # print(out)
                for i in range(0, 6):
                    oneStatus = prenew[i * 3:(i + 1) * 3]
                    out[i * 3:(i + 1) * 3] = 1 * torch.softmax(oneStatus, 0)
                    for index, value in enumerate(out[i * 3:(i + 1) * 3]):
                        if index < 3:
                            if value < 0.1:
                                out[i * 3 + index] = 0.1
                out = out.tolist()
                # probli = []
                consider = {}
                max_consider = []
                max_problis = []

                for move in availables:
                    move_p = move.pPawn
                    move_to = move.pMove
                    problis = out[(move_p - 1) * 3:move_p * 3]  # move = ['right', 'down', 'rightdown']
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
                # move_choose = choice(max_consider)
                Qsr[move_choose] = valuenet(datanew)[0][0]
                visited_red.add(move_choose)

                # 激活已选节点在playsr中，winsr中
                if playsr.get(move_choose, -1) == -1:
                    playsr[move_choose] = 0
                    winsr[move_choose] = 0
                visited_states.add(move_choose)
                k = 0
                if move_choose is not False:
                    k = isEnd(move_choose)
                    if k:
                        break

                # print(np.array(move_choose.map))
                # NL = []
                # NL.append(move_choose)
                # availables = getTheNextStepStatus(NL)  # 更新合法后继
                # del NL
                """
                    此处是要更新新的局面数据，以传入网络进行估值
                """

                """
                    把移动过的点加入进访问元组中，方便日后查询
                """

    for move in visited_states:
        playsr[move] += 1
        if k == 1:
            winsr[move] += 1
        if move in visited_red:
            if recorder.get(The_total_choose, 0) == 0:
                recorder[The_total_choose] = 0
            recorder[The_total_choose] += 1
            if k == 1:
                Qsr[move] = (Qsr.get(move, 0) + 1) / 2
            if k == 2:
                Qsr[move] = (Qsr.get(move, 0) - 1) / 2
            Vsr_all[The_total_choose] += Qsr.get(move, 0)
        Vsr[The_total_choose] = Vsr_all.get(The_total_choose, 0) / recorder.get(The_total_choose, 1)
    del visited_states
    del visited_red
    return playsr, winsr


def SoftMax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def playGame(Red, Blue, detail, now=0):  # 选择策略
    # global tr, tb
    global recordplay
    global recordtime
    global tmpTime, myTime, yourTime
    movered_moveblue = {'right': 'left', 'down': 'up', 'rightdown': 'leftup'}
    coldict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    lastInfo = []
    S.indx = 0
    mapNeedRedraw = True  # 是否需要重新绘制地图
    if detail:
        drawGameScreen(Red, Blue)

    cnt = 1  # 回合数
    myTime = yourTime = 0  # 时间重置
    while True:
        moveTo = None  # 棋子移动方向
        mapNeedRedraw = False
        s = '己方'
        if cnt % 2 == 0 and playhand == 'first' or cnt % 2 == 1 and playhand == 'second':
            s = '对方'
        cnt += 1
        number = input('请输入' + s + '的色子数: ')
        number = int(number)
        # n, ans = selectPawn(S)
        n, ans = selectPawnnewone(S, number)  # 2023-8-4
        # print('The number is ', n)
        if detail:
            drawMovePawn(n, ans)
            for event in pygame.event.get():
                if event.type == QUIT:
                    terminate()
        # t1 = time.time()
        mymap = np.array(S.map)
        if playhand == 'second':
            e_1 = np.identity(mymap.shape[0], dtype=np.int8)[:,::-1]
            mymap = e_1.dot(mymap).dot(e_1)
        print(mymap)
        tmpTime = time.time()  # 计时开始
        if COUNT % 2 == 0:
            # mymap = np.array(S.map)
            # print(S.map)
            # if playhand == 'second':
            #     e_1 = np.identity(mymap.shape[0], dtype=np.int8)[:,::-1]
            #     mymap = e_1.dot(mymap).dot(e_1)
            # print(mymap)
            if Red == 'Human':
                p, moveTo = decideRedHowToMove(ans)
            if Red == 'GoAhead':
                p, moveTo = redByBraveOfMan(ans)
            if Red == 'BetaCat1.0':
                p, moveTo = redByMinimax(ans)
            if Red == 'BetaCat2.0':
                p, moveTo = redByBeyes(ans, lastInfo)
            if Red == 'Uct':
                p, moveTo = redByUct(ans)
            if Red == 'Nerual_Uct':
                p, moveTo = redByNeuralUCT(ans)
                # redByMinimax(ans)
                myTime += time.time() - tmpTime
                if playhand == 'first':
                    print('告诉对方：', p, moveTo)
                if playhand == 'second':
                    print('告诉对方：', p, movered_moveblue[moveTo])
            if Red == 'UctPlusDemo':
                p, moveTo = redByUctPlusDemo(ans)
                redByMinimax(ans)
            if Red == 'WinnerAlgorithm':
                p, moveTo = winner(ans)
                redByMinimax(ans)
            if Red == 'NerualPure':
                p, moveTo = redByNerualpure(ans)
            if Red == 'NerualClassify':
                p, moveTo = redByNerualpurewithclassify(ans)
            if Red == 'ValuePure':
                p, moveTo = redByValuePure(ans)
            if Red == 'PolicyPure':
                p, moveTo = redByPolicypure(ans)
        if COUNT % 2 == 1:
            # mymap = np.array(S.map)
            # print(S.map)
            # print(mymap)
            if Blue == 'Human':
                # p, moveTo = decideBlueHowToMove(ans)
                p, moveTo = decideBlueHowToMove2(ans)
            if Blue == 'GoAhead':
                p, moveTo = blueByBraveOfMan(ans)
            if Blue == 'Demo':
                p, moveTo = blueByDemo(ans)
            if Blue == 'Demo2':
                p, moveTo = blueByDemo2(ans)
            if Blue == 'Uct':
                p, moveTo = blueByUct(ans)
            if Blue == 'Uct1':
                p, moveTo = blueByUct_z_prune(ans)
            if Blue == 'BetaCat1.0':
                p, moveTo = blueByMinimax(ans)

        # 输出时间
        print('己方用时：', round(myTime, 2), 's')
        print('对方用时：', round(yourTime, 2), 's')

        if moveTo != None:
            moved = makeMove(p, moveTo)
            # BEGIN: Added on 2023-8-4
            recordtime = recordtime + 1
            if playhand == 'first':
                if COUNT % 2 == 0:
                    row, col = getLocation(p, S.map)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'R' + str(p) + ',' + mycol + str(myrow) + ')')
                if COUNT % 2 == 1:
                    row, col = getLocation(p, S.map)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'B' + str(p - 6) + ',' + mycol + str(myrow) + ')')
            if playhand == 'second':
                if COUNT % 2 == 1:
                    secondmap = reverse(np.array(S.map))
                    secondmap = secondmap.tolist()
                    row, col = getLocation(p, secondmap)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'R' + str(p - 6) + ',' + mycol + str(myrow) + ')')
                if COUNT % 2 == 0:
                    secondmap = reverse(np.array(S.map))
                    secondmap = secondmap.tolist()
                    row, col = getLocation(p, secondmap)
                    myrow = 5 - row
                    mycol = coldict[col]
                    recordplay.append(
                        str(recordtime) + ':' + str(number) + ';(' + 'B' + str(p) + ',' + mycol + str(myrow) + ')')
            # END: Added on 2023-8-4
            S.indx += 1
            # t2 = time.time() - t1
            # if COUNT % 2 == 0:
            #     tr = tr + t2
            # else:
            #     tb = tb + t2
            lastInfo = [n, p, moveTo]
            if moved:
                mapNeedRedraw = True
                if mapNeedRedraw and detail:  # 如果需要重绘棋局界面，则：
                    drawGameScreen(Red, Blue)  # 重绘棋局界面
                    pass
        result = isEnd(S)  # 检查游戏是否结束，返回游戏结果
        if result:
            lastInfo = []
            return result


if __name__ == '__main__':
    '''
    可选测试对象
    Red  ：BetaCat1.0 | BetaCat2.0 | Human | GoAhead | Uct | Nerual_Uct | UctPlusDemo | WinnerAlgorithm | NerualPure
          NerualClassify | ValuePure | PolicyPure
    Blue ：BetaCat1.0 | GoAhead | Human | Demo | Demo2 | Uct 
    Human表示棋手为人类.
    '''
    mp.set_start_method('spawn')  # 多进程启动方法
    # data = open(r'C:\Users\Elessar\Desktop\Game_theory\chess\data.csv', 'a+', encoding='utf-8_sig')
    fiednames = ['map', 'UCT', 'winner']
    # Red = 'Nerual_Uct'
    Red = 'Nerual_Uct'
    Blue = 'Human'
    # filename = os.getcwd() + "/data/" + Red + 'Vs' + Blue + '.txt'
    # file = open('record_convergence_Neural.txt', 'w')

    # global all_data, all_label, record, we_result

    # all_data = np.load('data/input1.npy')
    all_data = np.array([])
    all_label = np.array([])
    record = np.array([])
    we_result = np.array([])
    all_data = all_data.tolist()
    all_label = all_label.tolist()
    record = record.tolist()
    we_result = we_result.tolist()

    # 注意网络层数对应的定义
    device = torch.device("cuda")
    # net = ResNet(BasicBlock, [1, 1, 1, 1])
    # net = net.cuda(device)
    # net.load_state_dict(torch.load('0830_1111_Alpha1.pt'))
    # net.eval()

    # valuenet = ResNetValue(BasicBlock, [1, 1, 1, 1], 1)
    # valuenet = valuenet.cuda(device)
    # valuenet.load_state_dict(torch.load('0830_1111_Alpha1_value_little.pt'))
    # valuenet.eval()

    temperature = 0.1
    temperature_demo = 2.0
    cnt = 7
    allcnt = cnt
    result = startGame(Red, Blue, cnt, False)

# REDWIN = 1  # 代表RED方赢
# BLUEWIN = 2  # 代表玩家BLUE方赢
# redByUctPlusDemo
