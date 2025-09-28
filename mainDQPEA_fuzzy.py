# coding:utf-8
import copy
import numpy as np
import os
import math
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from abc import ABC
from DataRead import DataReadDHFJSP
from Initial import GHInitial
from inital import initial
from fitFJSP import CalfitDHFJFP
from Tselection import *
from EA import evolution
from Tool import *
from FastNDSort import FastNDS
from EnergySave import EnergysavingDHFJSP
from LocalSearch import *
from DQN_model import DQN

def DataReadDHFJSP(Filepath):
    try:
        with open(Filepath, "r", encoding='utf-8') as f1:
            lines = f1.readlines()
    except FileNotFoundError:
        print(f"文件 {Filepath} 未找到")
        sys.exit(1)
    N, F, TM = map(int, lines[0].split())
    H = [0] * N
    op_list = []
    line_idx = 1
    f = 0
    # 初始化数组
    opmax = 5  # 初始假设，之后更新
    time = np.zeros((F, N, opmax, TM, 3), dtype=int)
    NM = np.zeros((F, N, opmax), dtype=int)
    M = np.zeros((F, N, opmax, TM), dtype=int)
    ProF = np.zeros((N, F))
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        line_idx += 1
        if not line:
            continue
        tokens = re.findall(r'(\d+)\s*\((\d+,\d+,\d+)\)', line)  # 匹配 "m (l,m,u)" 模式
        if not tokens:
            # 尝试解析工厂/作业/工序数
            tokens = re.findall(r'\d+', line)
            if len(tokens) == 3 and all(t.isdigit() for t in tokens):
                f = int(tokens[0]) - 1
                j = int(tokens[1]) - 1
                H[j] = int(tokens[2])
                op_list.append(H[j])
                continue
            continue
        # 解析工序和机器信息
        if 'o' not in locals():
            o = 0  # 假设第一行是工序 1
        else:
            o += 1  # 递增工序号
        NM_o = len(tokens)  # 可用机器数
        if NM_o > 0:
            for m_idx, (machine, fuzzy_str) in enumerate(tokens):
                machine = int(machine) - 1  # 机器号从 0 开始
                l, m_val, u = map(int, fuzzy_str.strip('()').split(','))
                if f < F and j < N and o < opmax and machine < TM:
                    time[f][j][o][machine] = [l, m_val, u]
                    M[f][j][o][machine] = machine + 1  # 机器索引 (1-based)
                    NM[f][j][o] = NM_o  # 记录可用机器数
    opmax = max(op_list) if op_list else 5
    # 确保数组维度正确
    if opmax > time.shape[2]:
        # 动态扩展 time 和 NM, M
        new_time = np.zeros((F, N, opmax, TM, 3), dtype=int)
        new_NM = np.zeros((F, N, opmax), dtype=int)
        new_M = np.zeros((F, N, opmax, TM), dtype=int)
        new_time[:F, :N, :min(opmax, time.shape[2]), :TM, :] = time
        new_NM[:F, :N, :min(opmax, NM.shape[2])] = NM
        new_M[:F, :N, :min(opmax, M.shape[2]), :TM] = M
        time, NM, M = new_time, new_NM, new_M
    SH = sum(H)  # 总工序数
    return N, F, TM, H, SH, NM, M, time, ProF

def initial(N, H, SH, NM, M, ps, F):
    p_chrom = np.zeros(shape=(ps, SH), dtype=int)
    m_chrom = np.zeros(shape=(ps, SH), dtype=int)
    f_chrom = np.zeros(shape=(ps, N), dtype=int)
    chrom = np.zeros(SH, dtype=int)
    FC = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(int(H[i])):
            k = int(sum(H[:i]) + j)
            chrom[k] = i
        FC[i] = i % F
    tmp = chrom
    tmp2 = FC
    random.shuffle(tmp)
    random.shuffle(tmp2)
    p_chrom[0, :] = tmp
    f_chrom[0, :] = tmp2
    for i in range(1, ps):
        tmp = p_chrom[i-1, :].copy()
        random.shuffle(tmp)
        p_chrom[i, :] = tmp
        tmp2 = f_chrom[i-1, :].copy()
        random.shuffle(tmp2)
        f_chrom[i, :] = tmp2
    for k in range(ps):
        for i in range(N):
            curf = f_chrom[k, i]
            for j in range(int(H[i])):
                t = int(math.floor(random.random() * NM[curf][i][j]))
                k1 = int(sum(H[:i]))
                t1 = int(k1 + j)
                m_chrom[k, t1] = M[curf][i][j][t] - 1
    return p_chrom, m_chrom, f_chrom

def RFA(subps,N,F): #随机分配工厂编号
    f_chrom = np.zeros(shape=(subps, N), dtype=int) #subps:子种群大小
    FC = np.zeros(N, dtype=int)
    # generate operation sequence randomly
    for i in range(N):
        FC[i] = i % F
    tmp2 = FC
    random.shuffle(tmp2)
    f_chrom[0, :] = tmp2
    for i in range(1,subps):
        tmp2 = f_chrom[i - 1, :]
        random.shuffle(tmp2)
        f_chrom[i, :] = tmp2
    return f_chrom

def RMS(subps,N,H,SH,NM,M): #随机选择机器
    m_chrom = np.zeros(shape=(subps, SH), dtype=int)
    # start generating machine selection vector
    for k in range(subps):
        for i in range(N):
            for j in range(int(H[i])):
                # 为每个操作随机选择一台机器，并填充到 m_chrom 中
                t = int(math.floor(random.random() * NM[i][j]))
                k1 = 0
                for t2 in range(i):
                    k1 = k1 + H[t2]
                t1 = int(k1 + j)
                m_chrom[k, t1] = M[i][j][t] - 1  # adjust the index from 0 to TM-1 to suit time
    return m_chrom

def MinPTF(subps,N,F,ProF): #最小处理时间工厂分配
    f_chrom = np.zeros(shape=(subps, N), dtype=int)
    for k in range(subps):
        #初始化工厂负载计数和工厂分配
        countf = np.zeros(F, dtype=int)
        Fchrom = np.zeros(N, dtype=int)
        for i in range(N):
            pro = ProF[i,:];
            pro_index= np.argsort(pro);
            #对每个任务的处理时间排序，选择最小处理时间的工厂。
            jpf = np.mean(countf);
            for f in range(F):
                if countf[pro_index[f]] <= jpf:
                    of = f;
                    break;
            Fchrom[i]= of;
            countf[of] = countf[of] + 1;
        f_chrom[k,:]=copy.copy(Fchrom); #将工厂分配结果填充到染色体中
    return f_chrom

def MinPTM(f_chrom, subps, N, H, SH, NM, time, TM, M):
    m_chrom = np.zeros((subps, SH), dtype=int)
    for k in range(subps):
        for i in range(N):
            curf = f_chrom[k][i]
            for j in range(int(H[i])):
                t1 = i
                t2 = j
                t4 = int(sum(H[:t1]))
                mp = t4 + t2
                index = M[curf][i][j][0] - 1
                z = time[curf][i][j][index]
                eta_z = (z[0] + 2 * z[1] + z[2]) / 4
                NM1 = int(NM[curf][i][j])
                for t in range(NM1):
                    d = M[curf][i][j][t] - 1
                    z_t = time[curf][i][j][d]
                    eta_t = (z_t[0] + 2 * z_t[1] + z_t[2]) / 4
                    if eta_z > eta_t or (eta_z == eta_t and z[1] > z_t[1]):
                        eta_z = eta_t
                        index = d
                m_chrom[k][mp] = index
    return m_chrom


def MinFTM(p_chrom, f_chrom, subps, N, H, SH, NM, time, TM, M, F):
    opmax = int(max(H))
    m_chrom = np.zeros((subps, SH), dtype=int)
    for k in range(subps):
        s1 = copy.copy(p_chrom[k, :])
        s2 = np.zeros(SH, dtype=int)
        p = np.zeros(N, dtype=int)
        for i in range(SH):
            p[s1[i]] += 1
            s2[i] = p[s1[i]]
        P = [[] for _ in range(F)]
        for i in range(SH):
            t1 = s1[i]
            t3 = f_chrom[k][t1]
            P[t3].append(p_chrom[k][i])
        for f in range(F):
            mt = np.zeros((TM, 3))  # 模糊负载 [l, m, u]
            SH1 = len(P[f])
            mm = np.zeros(SH1, dtype=int)
            s3 = copy.copy(P[f])
            s4 = np.zeros(SH1, dtype=int)
            p = np.zeros(N, dtype=int)
            finish = np.zeros((N, opmax, 3))
            for i in range(SH1):
                p[s3[i]] += 1
                s4[i] = p[s3[i]]
            for i in range(SH1):
                t1 = s3[i]
                t2 = s4[i] - 1
                MachineIndex = FindMinFinishTimeMachine(t1, t2, mt[:, 1], TM, N, M[f], NM[f])
                MinPT = FindMinProcessTimeMachine(t1, t2, MachineIndex, f, time, M[f])
                mm[i] = MinPT if isinstance(MinPT, int) else MinPT[0]
                z = time[f][t1][t2][mm[i]]
                prev_finish = finish[t1][t2 - 1] if s4[i] > 1 else [0, 0, 0]
                machine_available = mt[mm[i]]
                start = [max(prev_finish[0], machine_available[0]),
                         max(prev_finish[1], machine_available[1]),
                         max(prev_finish[2], machine_available[2])]
                finish[t1][t2] = [start[0] + z[0], start[1] + z[1], start[2] + z[2]]
                mt[mm[i]] = finish[t1][t2]
    return m_chrom

def MinWLM(p_chrom,f_chrom,subps,N,H,SH,NM,time,TM,M,F): #最小工作负载机器选择
    m_chrom = np.zeros(shape=(subps, SH), dtype=int)
    for k in range(subps):
        s1 = copy.copy(p_chrom[k, :])
        s2 = np.zeros(SH, dtype=int)
        p = np.zeros(N, dtype=int)
        for i in range(SH):
            p[s1[i]] = p[s1[i]] + 1
            s2[i] = p[s1[i]]
        P = [[] for _ in range(F)]
        # assign the machine to each operation from machine selection vector
        for i in range(SH):
            t1 = s1[i]
            t3 = f_chrom[k][t1]
            P[t3].append(p_chrom[k][i])

        for f in range(F):
            mt = np.zeros((TM, 3))  # Store fuzzy load as (a, b, c)
            SH1 = len(P[f])
            mm = np.zeros(SH, dtype=int)
            s3 = copy.copy(P[f])
            s4 = np.zeros(SH1, dtype=int)
            p = np.zeros(N, dtype=int)
            finish = np.zeros((N, max(H), 3))  # Store fuzzy finish times

            for i in range(SH1):
                p[s3[i]] = p[s3[i]] + 1
                s4[i] = p[s3[i]]
            for i in range(SH1):
                t1 = s3[i]
                t2 = s4[i] - 1
                n = NM[f][t1][t2]
                if n == 1:
                    mm[i] = M[f][t1][t2][0] - 1
                    z = time[f][t1][t2][mm[i]]
                    mt[mm[i]] = [mt[mm[i]][0] + z[0], mt[mm[i]][1] + z[1], mt[mm[i]][2] + z[2]]
                    continue
                else:
                    avalible_m = np.zeros(n, dtype=int)
                    avalible_m_load = np.zeros(n)
                    for j in range(n):
                        avalible_m[j] = M[f][t1][t2][j] - 1
                        avalible_m_load[j] = (mt[avalible_m[j]][0] + 2 * mt[avalible_m[j]][1] + mt[avalible_m[j]][2]) / 4
                    candidateM = selectMachine(avalible_m_load)
                    sizeM = len(candidateM)
                    if sizeM == 1:
                        mm[i] = avalible_m[candidateM[0]]
                        z = time[f][t1][t2][mm[i]]
                        mt[mm[i]] = [mt[mm[i]][0] + z[0], mt[mm[i]][1] + z[1], mt[mm[i]][2] + z[2]]
                    else:
                        eta_t = (time[f][t1][t2][avalible_m[candidateM[0]]][0] +
                                 2 * time[f][t1][t2][avalible_m[candidateM[0]]][1] +
                                 time[f][t1][t2][avalible_m[candidateM[0]]][2]) / 4
                        mm[i] = avalible_m[candidateM[0]]
                        for kk in range(1, sizeM):
                            eta_tmp = (time[f][t1][t2][avalible_m[candidateM[kk]]][0] +
                                       2 * time[f][t1][t2][avalible_m[candidateM[kk]]][1] +
                                       time[f][t1][t2][avalible_m[candidateM[kk]]][2]) / 4
                            if eta_t > eta_tmp or (eta_t == eta_tmp and
                                                   time[f][t1][t2][avalible_m[candidateM[kk]]][1] >
                                                   time[f][t1][t2][avalible_m[candidateM[0]]][1]):
                                mm[i] = avalible_m[candidateM[kk]]
                                eta_t = eta_tmp
                        z = time[f][t1][t2][mm[i]]
                        mt[mm[i]] = [mt[mm[i]][0] + z[0], mt[mm[i]][1] + z[1], mt[mm[i]][2] + z[2]]
            for i in range(SH1):
                t1 = s3[i]
                t2 = s4[i]
                t4 = 0
                for kk in range(t1):  # sum from 0 to t1-1
                    t4 = t4 + H[kk]
                mp = t4 + t2 - 1
                m_chrom[k][mp] = mm[i]
    return m_chrom

def selectMachine(mt):#输入机器负载向量 返回最小的机器负载 索引 可能是大于等于1
    L = len(mt)
    candidateM = []
    f = mt[0]
    index = 0
    for i in range(1, L):
        if f > mt[i]:
            f = mt[i]
            index = i
    candidateM.append(index)
    f = mt[index]
    for i in range(L):
        if i == index:
            continue
        elif f == mt[i]:
            candidateM.append(i)
    return candidateM

def FindMinFinishTimeMachine(JobIndex, OperationIndex, mt, TM, N, M, NM):
    L = NM[JobIndex][OperationIndex]
    CandidateM = np.zeros(L, dtype=int)
    for i in range(L):
        CandidateM[i] = M[JobIndex][OperationIndex][i] - 1
    if L < 2:
        return CandidateM[0]  # 返回单一机器
    MinFinishMIndex = CandidateM[0]
    eta_min = (mt[MinFinishMIndex][0] + 2 * mt[MinFinishMIndex][1] + mt[MinFinishMIndex][2]) / 4
    for j in range(1, L):
        eta_j = (mt[CandidateM[j]][0] + 2 * mt[CandidateM[j]][1] + mt[CandidateM[j]][2]) / 4
        if eta_min > eta_j:
            MinFinishMIndex = CandidateM[j]
            eta_min = eta_j
    return MinFinishMIndex

def FindMinProcessTimeMachine(JobIndex, OperationIndex, MachineIndex, f_index, time, M):
    L = len(MachineIndex)
    if L < 2:
        return MachineIndex[0]
    MinProcessTimeMachine = MachineIndex[0]
    z = time[f_index][JobIndex][OperationIndex][MinProcessTimeMachine]
    eta_z = (z[0] + 2 * z[1] + z[2]) / 4
    for i in range(1, L):
        z_t = time[f_index][JobIndex][OperationIndex][MachineIndex[i]]
        eta_t = (z_t[0] + 2 * z_t[1] + z_t[2]) / 4
        if eta_z > eta_t or (eta_z == eta_t and z_t[1] > z[1]):
            MinProcessTimeMachine = MachineIndex[i]
            eta_z = eta_t
            z = z_t
    return MinProcessTimeMachine

def GHInitial(N,H,SH,NM,M,TM,time,F,ProF,ps): #结合多种启发式方法初始化染色体
    p_chrom = np.zeros(shape=(ps, SH), dtype=int)
    m_chrom = np.zeros(shape=(ps, SH), dtype=int)
    f_chrom = np.zeros(shape=(ps, N), dtype=int)

    chrom = np.zeros(SH, dtype=int)
    # generate operation sequence randomly
    for i in range(N):
        for j in range(int(H[i])):
            k = 0
            for t in range(i - 1):
                k = k + H[t]
            k = int(k + j)
            chrom[k] = i

    tmp = chrom;
    random.shuffle(tmp)
    p_chrom[0, :] = copy.copy(tmp)

    for i in range(1, ps):
        tmp = p_chrom[i - 1, :]
        random.shuffle(tmp)
        p_chrom[i, :] = copy.copy(tmp)

    subps=math.floor(ps/5) #将种群分为 5 个子种群，每个子种群应用不同的初始化方法
    for i in range(5):
        low = (i) * subps;
        up = low + subps;
        subpchrom = p_chrom[low:up,:];
        print(i)
        if i==0: #第一个子种群使用最小处理时间工厂分配和随机机器选择
            subfchrom = MinPTF(subps,N,F,ProF);
            submchrom = RMS(subps,N,H,SH,NM,M);  # 注意: RMS未修正，需要根据实际情况修正
        elif i==1: #第二个子种群使用随机工厂分配和最小处理时间机器选择
            subfchrom = RFA(subps,N,F);
            submchrom = MinPTM(subfchrom,subps,N,H,SH,NM,time,TM,M);
        elif i==2: #第三个子种群使用随机工厂分配和最小工作负载机器选择
            subfchrom = RFA(subps,N,F);
            submchrom = MinWLM(subpchrom,subfchrom,subps,N,H,SH,NM,time,TM,M,F);
        elif i==3: #第四个子种群使用最小完成时间机器选择和随机工厂分配
            submchrom = MinFTM(subpchrom,subfchrom,subps,N,H,SH,NM,time,TM,M,F);
            subfchrom = RFA(subps,N,F);
        elif i==4: #第五个子种群使用随机机器选择和随机工厂分配
            submchrom = RMS(subps,N,H,SH,NM,M);
            subfchrom = RFA(subps,N,F);
        m_chrom[low: up,:]=copy.copy(submchrom);
        f_chrom[low: up,:]=copy.copy(subfchrom);
    return p_chrom,m_chrom,f_chrom

def CalfitDHFJFP(p_chrom, m_chrom, f_chrom, N, H, SH, F, TM, time):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] += 1
        s2[i] = p[s1[i]]
    P = [[] for _ in range(F)]
    FJ = [[] for _ in range(F)]
    for i in range(SH):
        t1 = s1[i]
        t3 = f_chrom[t1]
        P[t3].append(s1[i])  # 注意 P[f] 存储任务 ID，而非 p_chrom[i]
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)
    sub_f_fit = np.zeros((F, 2))  # 每个工厂的 [makespan eta, energy eta]
    for f in range(F):
        if len(P[f]) > 0:  # 避免空工厂
            sub_f_fit[f][0], sub_f_fit[f][1] = CalfitFJFP(P[f], m_chrom, FJ[f], f, N, H, TM, time)
    fit1 = 0  # 最大完成时间 eta
    fit2 = 0  # 总能量消耗 eta
    fit3 = 0  # 完成时间最长的工厂索引
    for f in range(F):
        fit2 += sub_f_fit[f][1]
        if sub_f_fit[f][0] > fit1:
            fit1 = sub_f_fit[f][0]
            fit3 = f
    return fit1, fit2, fit3  # 最大完成时间 eta、总能量消耗 eta、完成时间最长的工厂索引

def CalfitFJFP(p_chrom, m_chrom, FJ, f_index, N, H, TM, time):
    SH = len(p_chrom)  # 总操作数量
    Ep = 4; Es = 1  # 运行功率与空闲功率
    opmax = int(max(H))
    # 初始化完成时间矩阵用于记录每个操作的完成时间
    finish = np.zeros((N, opmax, 3))
    # machine finish time
    mt = np.zeros((TM, 3))
    # sign all operation
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)  # 操作队列
    p = np.zeros(N, dtype=int)  # 任务计数
    fitness = np.zeros(2)
    for i in range(SH):  # 计算每个任务的操作计数
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    # assign the machine to each operation from machine selection vector
    mm = np.zeros(SH, dtype=int)
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t4 = 0
        for k in range(t1):  # sum from 0 to t1-1
            t4 = t4 + H[k]
        mm[i] = m_chrom[t4 + t2 - 1]
    # 初始化工作载荷与待机时间
    TWL = np.zeros(3)
    TIT = np.zeros(3)  # 计算每个操作的完成时间、总工作负载（TWL）和总空闲时间（TIT）
    # start decoding
    for i in range(SH):
        # because array index starts with 0 in python, number in M start from 1,
        # number in time start from 0, thus number of m_chrom need to minus 1
        t1 = s1[i]
        t2 = s2[i] - 1
        t3 = mm[i]
        z = time[f_index][t1][t2][t3]
        if s2[i] == 1:  # 第一个操作 (注意 s2[i]==1 而非0，索引调整)
            mt[t3] = [mt[t3][0] + z[0], mt[t3][1] + z[1], mt[t3][2] + z[2]]
            finish[t1][t2] = mt[t3].copy()
            TWL = [TWL[0] + z[0], TWL[1] + z[1], TWL[2] + z[2]]
        else:
            prev_finish = finish[t1][t2 - 1]
            machine_available = mt[t3]
            # idle_time = max(0, prev_finish - machine_available) 分量处理
            idle_time = [max(0, prev_finish[0] - machine_available[0]),
                         max(0, prev_finish[1] - machine_available[1]),
                         max(0, prev_finish[2] - machine_available[2])]
            TIT = [TIT[0] + idle_time[0], TIT[1] + idle_time[1], TIT[2] + idle_time[2]]
            # start = max(prev_finish, machine_available)
            start = [max(prev_finish[0], machine_available[0]),
                     max(prev_finish[1], machine_available[1]),
                     max(prev_finish[2], machine_available[2])]
            mt[t3] = [start[0] + z[0], start[1] + z[1], start[2] + z[2]]
            finish[t1][t2] = mt[t3].copy()
            TWL = [TWL[0] + z[0], TWL[1] + z[1], TWL[2] + z[2]]
    # 计算 max makespan eta
    max_eta = (mt[0][0] + 2 * mt[0][1] + mt[0][2]) / 4
    for i in range(1, TM):
        eta = (mt[i][0] + 2 * mt[i][1] + mt[i][2]) / 4
        if eta > max_eta:
            max_eta = eta
    fitness[0] = max_eta
    # 计算能量消耗 eta
    eta_TWL = (TWL[0] + 2 * TWL[1] + TWL[2]) / 4
    eta_TIT = (TIT[0] + 2 * TIT[1] + TIT[2]) / 4
    fitness[1] = eta_TWL * Ep + eta_TIT * Es
    return fitness[0], fitness[1]  # 最大完成时间 eta、能量消耗 eta

def SwapOF(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time):
    Index1 = np.floor(random.random() * SH)
    Index2 = np.floor(random.random() * SH)
    while Index1 == Index2:
        Index2 = np.floor(random.random() * SH)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1); Index2 = int(Index2)
    tmp = copy.copy(newp[Index1])
    newp[Index1] = copy.copy(newp[Index2])  # 交换两点的染色体值
    newp[Index2] = copy.copy(tmp)
    return newp, newm, newf

# 在关键工厂（花费时间最长的工厂索引）中随机交换两个操作
def SwapIF(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    IP = []
    FJ = []
    for f in range(F):
        P.append([])
        IP.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        IP[t3].append(i)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])  # 寻找关键工厂
    L = len(IP[cf])
    Index1 = np.floor(random.random() * L)
    Index2 = np.floor(random.random() * L)
    while Index1 == Index2:
        Index2 = np.floor(random.random() * L)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1); Index2 = int(Index2)
    Index1 = IP[cf][Index1]; Index2 = IP[cf][Index2]
    tmp = copy.copy(newp[Index1])
    newp[Index1] = copy.copy(newp[Index2])  # 交换两点的染色体值
    newp[Index2] = copy.copy(tmp)
    return newp, newm, newf

# 插入操作
def InsertOF(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time):
    Index1 = np.floor(random.random() * SH)
    Index2 = np.floor(random.random() * SH)
    while Index1 == Index2:
        Index2 = np.floor(random.random() * SH)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1); Index2 = int(Index2)
    low = min(Index1, Index2)
    up = max(Index1, Index2)
    tmp = newp[up]
    for i in range(up, low, -1):
        newp[i] = copy.copy(newp[i - 1])
    newp[low] = copy.copy(tmp)
    return newp, newm, newf

# 关键工厂插入
def InsertIF(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    IP = []
    FJ = []
    for f in range(F):
        P.append([])
        IP.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        IP[t3].append(i)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])  # 寻找关键工厂
    L = len(IP[cf])
    Index1 = np.floor(random.random() * L)
    Index2 = np.floor(random.random() * L)
    while Index1 == Index2:
        Index2 = np.floor(random.random() * L)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1); Index2 = int(Index2)
    Index1 = IP[cf][Index1]; Index2 = IP[cf][Index2]
    low = min(Index1, Index2)
    up = max(Index1, Index2)
    tmp = newp[up]
    for i in range(up, low, -1):
        newp[i] = copy.copy(newp[i - 1])
    newp[low] = copy.copy(tmp)
    return newp, newm, newf

# 随机工厂分配
def RandFA(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F):
    Index1 = np.floor(random.random() * N)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1)
    newf[Index1] = int(np.floor(random.random() * F))
    return newp, newm, newf

# 基于加工能力分配工厂
def RankFA(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F, ProF):
    Index1 = np.floor(random.random() * N)
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(Index1)
    pro = ProF[Index1, :]
    pro_index = np.argsort(pro)
    x = random.random()
    n = len(pro)
    for f in range(n):
        if x < pro[f]:
            newf[Index1] = pro_index[f]
            break
    return newp, newm, newf

# 随机机器选择
def RandMS(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F):
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(np.floor(random.random() * SH))
    t1 = p_chrom[Index1]
    t4 = 0
    for k in range(t1):
        t4 = t4 + H[k]
    t2 = Index1 - t4 + 1
    curf = f_chrom[t1]
    NM1 = NM[curf][t1][t2 - 1]
    t = int(np.floor(random.random() * NM1))
    newm[Index1] = M[curf][t1][t2 - 1][t] - 1
    return newp, newm, newf

# 基于处理时间选择机器
def RankMS(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F):
    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    Index1 = int(np.floor(random.random() * SH))
    t1 = p_chrom[Index1]
    t4 = 0
    for k in range(t1):
        t4 = t4 + H[k]
    t2 = Index1 - t4 + 1
    curf = f_chrom[t1]

    print(f"t1: {t1}, t2: {t2}, NM shape: {np.array(NM).shape}, H: {H}")
    if t1 >= len(NM):
        raise ValueError(f"t1 ({t1}) out of bounds for NM with length {len(NM)}")
    if t2 - 1 >= len(NM[t1]):
        raise ValueError(f"t2-1 ({t2 - 1}) out of bounds for NM[{t1}] with length {len(NM[t1])}")

    NM1 = NM[curf][t1][t2 - 1]
    cm = np.zeros(NM1, dtype=int)
    ct = np.zeros(NM1)
    for t in range(NM1):
        d = M[curf][t1][t2 - 1][t] - 1
        z = time[curf][t1][t2 - 1][d]
        ct[t] = (z[0] + 2 * z[1] + z[2]) / 4  # 使用模糊数 eta 值进行排序
        cm[t] = d
    pro_index = np.argsort(ct)
    pro = np.zeros(NM1)
    for t in range(NM1):
        pro[t] = (NM1 - t) / (NM1 * (NM1 + 1) / 2)
    for t in range(1, NM1):
        pro[t] = pro[t] + pro[t - 1]
    x = random.random()
    tmp = newm[Index1]
    for t in range(NM1):
        if x < pro[t]:
            newm[Index1] = cm[pro_index[t]]
            break
    while tmp == newm[Index1]:
        x = random.random()
        for t in range(NM1):
            if x < pro[t]:
                newm[Index1] = cm[pro_index[t]]
                break
    return newp, newm, newf

# 根据关键块信息调整操作序列
def N6(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F):
    s1 = p_chrom  # 当前操作序列
    s2 = np.zeros(SH, dtype=int)  # 每个任务的操作计数
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    IP = []
    FJ = []
    for f in range(F):
        P.append([])
        IP.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        IP[t3].append(i)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])  # 关键工厂的索引
    CP, CB, block = FindCriticalPathDHFJSP(P[cf], m_chrom, FJ[cf], cf, N, H, time, TM)
    # 关键路径、关键块列表、关键块数量
    for i in range(block):
        BL = len(CB[i])
        if BL > 1:
            if i == 0:  # 第一个关键块，随机选择两个操作交换
                Index1 = int(np.floor(random.random() * (BL - 1)))
                Index2 = BL - 1
                Index1 = CB[i][Index1]; Index2 = CB[i][Index2]
                tmp = P[cf][Index1]
                for j in range(Index1, Index2):
                    P[cf][j] = P[cf][j + 1]
                P[cf][Index2] = tmp
            if i == block - 1:  # 最后一块，随机选择两个操作进行插入
                Index1 = 0
                Index2 = int(np.floor(random.random() * (BL - 1)) + 1)
                Index1 = CB[i][Index1]; Index2 = CB[i][Index2]
                tmp = P[cf][Index2]
                for j in range(Index2, Index1, -1):
                    P[cf][j] = P[cf][j - 1]
                P[cf][Index1] = tmp
            if i > 0 and i < block - 1 and BL > 2:  # 中间的关键块随机选择两个操作进行交换和插入
                Index1 = int(np.floor(random.random() * (BL - 2)) + 1)
                Index2 = BL - 1
                Index1 = CB[i][Index1]; Index2 = CB[i][Index2]
                tmp = P[cf][Index1]
                for j in range(Index1, Index2):
                    P[cf][j] = P[cf][j + 1]
                P[cf][Index2] = tmp
                Index1 = 0
                Index2 = int(np.floor(random.random() * (BL - 2)) + 1)
                Index1 = CB[i][Index1]; Index2 = CB[i][Index2]
                tmp = P[cf][Index2]
                for j in range(Index2, Index1, -1):
                    P[cf][j] = P[cf][j - 1]
                P[cf][Index1] = tmp
    newm = m_chrom; newf = f_chrom
    newp = np.zeros(SH, dtype=int)
    for f in range(F):
        L = len(IP[f])
        for i in range(L):
            newp[IP[f][i]] = P[f][i]
    return newp, newm, newf



class Net(nn.Module, ABC):
    def __init__(self, inDim, outDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inDim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.out = nn.Linear(32, outDim)

    def forward(self, x):
        # return self.out(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action_value = self.out(x)
        return action_value

class DQN(object):
    def __init__(self, inDim, outDim, BATCH_SIZE, LR , EPSILON, GAMMA, MEMORY_CAPACITY,TARGET_REPLACE_ITER):
        self.eval_net, self.target_net = Net(inDim, outDim), Net(inDim, outDim) #创建评估网络和目标网络
        # global N_STATES, N_ACTIONS
        self.N_STATES = inDim #输入
        self.N_ACTIONS = outDim #输出
        self.learn_step_counter = 0                                     # 学习步骤初始化为0
        self.memory_counter = 0                                         # 记录存储的经验数量
        self.BATCH_SIZE=BATCH_SIZE                                      # DQN超参数
        self.LR=LR
        self.EPSILON=EPSILON
        self.GAMMA=GAMMA
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  #使用 Adam优化器选择步长
        #self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR) #（SGD）随机梯度下降
        # memory是一个np数组，每一行代表一个记录，状态 动作 奖励 新的状态
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_STATES * 2 + 2))     #  初始化经验池
        #当前状态+新状态+动作+奖励
        self.loss_func = nn.MSELoss()
        #使用均方误差（MSE）作为损失函数，用于计算预测Q值（eval评估网络输出Q值）与目标Q值（target目标网络）之间的差异
        self.eval_net, self.target_net = self.eval_net.cuda(), self.target_net.cuda()
        self.loss_func = self.loss_func.cuda()
        #将评估网络、目标网络和损失函数移动到 GPU 上，以加速计算（还未开始运算）

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if torch.cuda.is_available():
            x = x.cuda()
        if np.random.uniform() < self.EPSILON:
            actions_value = self.eval_net.forward(x)
            if torch.cuda.is_available():
                actions_value = actions_value.cpu()
            actions_value = actions_value.detach().numpy()
            actions_value[actions_value <= 0] = 0.001
            actions_value = actions_value / np.sum(actions_value)
            max_v = np.max(actions_value[0])
            max_action = np.where(actions_value[0] == max_v)[0]
            action = max_action[random.randint(0, len(max_action) - 1)] if len(max_action) > 1 else max_action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_): #将经验放到经验回放缓冲区
        # 数组合并，a和r也新建个数组
        transition = np.hstack((s, [a, r], s_))
        #将当前的经验（状态 s、动作 a、奖励 r、新状态 s_）存储到经验回放缓冲区中
        # 缓冲区满新经验覆盖旧经验
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        if torch.cuda.is_available():
            b_s, b_a, b_r, b_s_ = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda()
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        losses = loss.cpu().detach().numpy()
        print('train loss MSE=', losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return losses

def tournamentSelection(p_chrom, m_chrom, f_chrom, fitness, ps, SH, N):
    # 锦标赛选择：通过随机选择两个个体进行比较，选择适应度较高的个体进入交配池
    # 初始化交配池
    pool_size = ps
    P_pool = np.zeros(shape=(ps, SH), dtype=int)
    M_pool = np.zeros(shape=(ps, SH), dtype=int)
    F_pool = np.zeros(shape=(ps, N), dtype=int)  # fitness of pool solutions
    # competitor number
    tour = 2

    # 随机从池子里选择两个个体
    for i in range(pool_size):
        index1 = int(math.floor(random.random() * ps))
        index2 = int(math.floor(random.random() * ps))
        while index1 == index2:
            index2 = int(math.floor(random.random() * ps))
        f1 = fitness[index1, 0:2]  # [makespan eta, energy eta]
        f2 = fitness[index2, 0:2]  # [makespan eta, energy eta]
        # 使用 NDS（非支配排序）计算两个个体适应度并比较
        if (NDS(f1, f2) == 1):  # 选个体 index1
            P_pool[i, :] = p_chrom[index1, :]
            M_pool[i, :] = m_chrom[index1, :]
            F_pool[i, :] = f_chrom[index1, :]
        elif (NDS(f1, f2) == 2):  # 选择个体 index2
            P_pool[i, :] = p_chrom[index2, :]
            M_pool[i, :] = m_chrom[index2, :]
            F_pool[i, :] = f_chrom[index2, :]
        else:  # 无法区分优劣，随机选择一个放入交配池
            if random.random() <= 0.5:
                P_pool[i, :] = p_chrom[index1, :]
                M_pool[i, :] = m_chrom[index1, :]
                F_pool[i, :] = f_chrom[index1, :]
            else:
                P_pool[i, :] = p_chrom[index2, :]
                M_pool[i, :] = m_chrom[index2, :]
                F_pool[i, :] = f_chrom[index2, :]

    print(f"i={i}, index1={index1}, index2={index2}")
    print(f"f1={f1}, f2={f2}, f1 shape={f1.shape}")
    return P_pool, M_pool, F_pool

def crossover(P1, M1, F1, P2, M2, F2, N, SH, F):
    # 初始化子代，复制父代个体作为子代初始值
    NP1 = P1.copy()
    NM1 = M1.copy()
    NF1 = F1.copy()
    NP2 = P2.copy()
    NM2 = M2.copy()
    NF2 = F2.copy()

    # 随机选择一部分任务 J1
    temp = [random.random() for _ in range(N)]
    temp = mylistRound(temp)  # 假设将随机数四舍五入到 0 或 1
    J1 = find_all_index(temp, 1)  # 找到值为 1 的任务索引

    # 初始化 ci1 和 ci2
    ci1 = np.zeros(SH, dtype=int)
    ci2 = np.zeros(SH, dtype=int)

    # 填充 ci1 和 ci2 的初始部分
    for j in range(SH):
        if Ismemeber(P1[j], J1) == 1:  # P1 保留 J1 中的操作
            ci1[j] = P1[j]
        if Ismemeber(P2[j], J1) == 0:  # P2 保留非 J1 中的操作
            ci2[j] = P2[j]

    # 找到空位和非空位
    index_1_1 = find_all_index(ci1, 0)  # ci1 中的空位
    index_1_2 = find_all_index_not(ci2, 0)  # ci2 中的非空位
    index_2_1 = find_all_index(ci2, 0)  # ci2 中的空位
    index_2_2 = find_all_index_not(ci1, 0)  # ci1 中的非空位

    l1 = len(index_1_1)
    l2 = len(index_2_1)
    if l1 == len(index_1_2) and l2 == len(index_2_2):  # 确保长度匹配
        for j in range(l1):
            ci1[index_1_1[j]] = P2[index_1_2[j]]  # 填充 ci1 的空位
        for j in range(l2):
            ci2[index_2_1[j]] = P1[index_2_2[j]]  # 填充 ci2 的空位

    # 更新 NP1 和 NP2
    NP1 = ci1
    NP2 = ci2

    # 对机器选择运用通用交叉
    s = [random.random() for _ in range(SH)]
    s = mylistRound(s)
    for i in range(SH):
        if s[i] == 1:
            NM1[i], NM2[i] = NM2[i], NM1[i]  # 交换机器选择

    # 对工厂分配运用通用交叉
    s = [random.random() for _ in range(N)]
    s = mylistRound(s)
    for i in range(N):
        if s[i] == 1:
            t = NF1[i]
            NF1[i] = NF2[i]
            NF2[i] = t
            if NF1[i] >= F or NF2[i] >= F:  # 边界检查
                NF1[i] = NF1[i] % F
                NF2[i] = NF2[i] % F

    return NP1, NM1, NF1, NP2, NM2, NF2

def mutation(p_chrom, m_chrom, SH, N, H, NM, M, f_chrom):
    # 随机交换操作序列中的两个位置
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)
    t = p_chrom[p1]
    p_chrom[p1] = p_chrom[p2]
    p_chrom[p2] = t

    # 随机改变机器选择中的两个位置
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        if s1[i] < N:  # 确保作业号有效
            p[s1[i]] = p[s1[i]] + 1
            s2[i] = p[s1[i]]
    s3 = m_chrom

    # 变异两个随机位置
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)

    # 第一个变异位置
    if p1 < SH and s1[p1] < N and s2[p1] - 1 < H[s1[p1]]:  # 边界检查
        f_idx = f_chrom[s1[p1]]  # 获取工厂号
        n = NM[f_idx][s1[p1]][s2[p1] - 1]  # 可用机器数
        if n > 0:
            m = math.floor(random.random() * n)
            x = M[f_idx][s1[p1]][s2[p1] - 1][m] - 1  # 新机器号 (0-based)
            # 确保新机器与原机器不同
            if n > 1:
                while s3[p1] == x:
                    m = math.floor(random.random() * n)
                    x = M[f_idx][s1[p1]][s2[p1] - 1][m] - 1
            m_chrom[p1] = x  # 直接更新 m_chrom[p1]

    # 第二个变异位置
    if p2 < SH and s1[p2] < N and s2[p2] - 1 < H[s1[p2]]:  # 边界检查
        f_idx = f_chrom[s1[p2]]  # 获取工厂号
        n = NM[f_idx][s1[p2]][s2[p2] - 1]  # 可用机器数
        if n > 0:
            m = math.floor(random.random() * n)
            x = M[f_idx][s1[p2]][s2[p2] - 1][m] - 1  # 新机器号 (0-based)
            # 确保新机器与原机器不同
            if n > 1:
                while s3[p2] == x:
                    m = math.floor(random.random() * n)
                    x = M[f_idx][s1[p2]][s2[p2] - 1][m] - 1
            m_chrom[p2] = x  # 直接更新 m_chrom[p2]

    return p_chrom, m_chrom

def crossover(P1, M1, F1, P2, M2, F2, N, SH, F):
    # 初始化子代，复制父代个体作为子代初始值
    NP1 = P1.copy()
    NM1 = M1.copy()
    NF1 = F1.copy()
    NP2 = P2.copy()
    NM2 = M2.copy()
    NF2 = F2.copy()

    # 随机选择一部分任务 J1
    temp = [random.random() for _ in range(N)]
    temp = mylistRound(temp)  # 假设将随机数四舍五入到 0 或 1
    J1 = find_all_index(temp, 1)  # 找到值为 1 的任务索引

    # 初始化 ci1 和 ci2
    ci1 = np.zeros(SH, dtype=int)
    ci2 = np.zeros(SH, dtype=int)

    # 填充 ci1 和 ci2 的初始部分
    for j in range(SH):
        if Ismemeber(P1[j], J1) == 1:  # P1 保留 J1 中的操作
            ci1[j] = P1[j]
        if Ismemeber(P2[j], J1) == 0:  # P2 保留非 J1 中的操作
            ci2[j] = P2[j]

    # 找到空位和非空位
    index_1_1 = find_all_index(ci1, 0)  # ci1 中的空位
    index_1_2 = find_all_index_not(ci2, 0)  # ci2 中的非空位
    index_2_1 = find_all_index(ci2, 0)  # ci2 中的空位
    index_2_2 = find_all_index_not(ci1, 0)  # ci1 中的非空位

    l1 = len(index_1_1)
    l2 = len(index_2_1)
    if l1 == len(index_1_2) and l2 == len(index_2_2):  # 确保长度匹配
        for j in range(l1):
            ci1[index_1_1[j]] = P2[index_1_2[j]]  # 填充 ci1 的空位
        for j in range(l2):
            ci2[index_2_1[j]] = P1[index_2_2[j]]  # 填充 ci2 的空位

    # 更新 NP1 和 NP2
    NP1 = ci1
    NP2 = ci2

    # 对机器选择运用通用交叉
    s = [random.random() for _ in range(SH)]
    s = mylistRound(s)
    for i in range(SH):
        if s[i] == 1:
            NM1[i], NM2[i] = NM2[i], NM1[i]  # 交换机器选择

    # 对工厂分配运用通用交叉
    s = [random.random() for _ in range(N)]
    s = mylistRound(s)
    for i in range(N):
        if s[i] == 1:
            t = NF1[i]
            NF1[i] = NF2[i]
            NF2[i] = t
            if NF1[i] >= F or NF2[i] >= F:  # 边界检查
                NF1[i] = NF1[i] % F
                NF2[i] = NF2[i] % F

    return NP1, NM1, NF1, NP2, NM2, NF2

def evolution2(p_chrom,m_chrom,f_chrom,index,T,neighbour,Pc,Pm,ps,SH,N,H,NM,M): #邻域选择两个个体交叉变异
    nei=neighbour[index,:] # 一个矩阵，其中每一行表示一个个体的邻域个体索引

    R1 = math.floor(random.random() * T)
    R1 = nei[R1]
    R2 = math.floor(random.random() * T)
    R1 = nei[R2]
    #R2 = nei[R2]
    P1=copy.copy(p_chrom[R1,:])
    P2 = copy.copy(p_chrom[R2,:])
    M1=copy.copy(m_chrom[R1,:])
    M2 = copy.copy(m_chrom[R2,:])
    F1=copy.copy(f_chrom[R1,:])
    F2 = copy.copy(f_chrom[R2,:])
    while R1 == R2:
        R2 = math.floor(random.random() * T)
        R2= nei[R2]
    if random.random()<Pc:
        P1,M1,F1,P2,M2,F2=crossover(p_chrom[R1,:],m_chrom[R1,:],f_chrom[R1,:],p_chrom[R2,:],m_chrom[R2,:],f_chrom[R2,:],N,SH)
    if random.random()<Pm:
        P1,M1=mutation(P1,M1,SH,N,H,NM,M)
        P2, M2 = mutation(P2, M2, SH, N, H, NM, M)
    return P1,M1,F1,P2,M2,F2

def mylistRound(arr):
    """将随机数数组根据 0.5 阈值二值化，返回新数组。"""
    return [1 if x > 0.5 else 0 for x in arr]

def find_all_index(arr, item):
    return np.where(np.array(arr) == item)[0].tolist()
def find_all_index_not(arr, item):
    return np.where(np.array(arr) != item)[0].tolist()

def NDS(fit1, fit2):
    # fit1 和 fit2 是 [makespan eta, energy eta] 的数组，形状 (2,)
    dom_less = 0
    dom_equal = 0
    dom_more = 0
    for k in range(2):  # 比较 makespan eta 和 energy eta
        if fit1[k] > fit2[k]:
            dom_more += 1
        elif fit1[k] == fit2[k]:
            dom_equal += 1
        else:
            dom_less += 1
    if dom_less == 0 and dom_equal != 2:
        return 2  # fit1 支配 fit2
    if dom_more == 0 and dom_equal != 2:
        return 1  # fit2 支配 fit1
    return 0  # 非支配

def Ismemeber(item, list):
    """检查 item 是否在 list 中，返回 1 (在) 或 0 (不在)。"""
    if not list:  # 空列表检查
        return 0
    return 1 if item in list else 0

def DeleteReapt(QP, QM, QF, QFit, ps):
    row = np.size(QFit, 0)
    i = 0
    while i < row:
        if i >= row:
            break

        F = QFit[i, :]
        j = i + 1
        while j < row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1]:
                QP = np.delete(QP, j, axis=0)
                QM = np.delete(QM, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j = j - 1
                row = row - 1
                if row < 2 * ps + 1:
                    break
            j = j + 1
        i = i + 1
        if row < 2 * ps + 1:
            break
    return QP, QM, QF, QFit

def DeleteReaptE(QP, QM, QF, QFit):  # for elite strategy
    row = np.size(QFit, 0)
    i = 0
    while i < row:
        if i >= row:
            break

        F = QFit[i, :]
        j = i + 1
        while j < row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1]:
                QP = np.delete(QP, j, axis=0)
                QM = np.delete(QM, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j = j - 1
                row = row - 1
            j = j + 1
        i = i + 1

    return QP, QM, QF, QFit

def pareto(fitness):
    PF = []
    L = np.size(fitness, axis=0)
    pn = np.zeros(L, dtype=int)
    for i in range(L):
        for j in range(L):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(2):  # number of objectives
                if (fitness[i][k] > fitness[j][k]):
                    dom_more = dom_more + 1
                elif (fitness[i][k] == fitness[j][k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1

            if dom_less == 0 and dom_equal != 2:  # i is dominated by j
                pn[i] = pn[i] + 1
        if pn[i] == 0:  # add i into pareto front
            PF.append(i)
    return PF

class Machine(object):
    def __init__(self):
        self.Op = []
        self.GapT = []  # 存储模糊间隙时间 [a, b, c]
        self.MFT = []  # 存储模糊完成时间 [a, b, c]

def SAS2AS(p_chrom, m_chrom, FJ, N, H, TM, time, f_index):  # 从前遍历
    SH = len(p_chrom)
    e = 0
    opmax = int(max(H))
    # 初始化，finish和start矩阵记录每个操作的模糊开始与完成时间
    finish = np.zeros(shape=(N, opmax, 3))
    start = np.zeros(shape=(N, opmax, 3))
    mt = np.zeros((TM, 3))  # 记录每个机器的当前模糊时间 [a, b, c]
    MA = []  # 是机器对象列表，用于跟踪每个机器上的操作、模糊空闲时间和模糊完成时间
    for i in range(TM):  # creat a object array
        MA.append(Machine())

    s1 = p_chrom  # 操作序列
    s2 = np.zeros(SH, dtype=int)  # 操作队列
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    mm = np.zeros(SH, dtype=int)  # 机器分配
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t4 = 0
        for k in range(t1):  # sum from 0 to t1-1
            t4 = t4 + H[k]
        mm[i] = m_chrom[t4 + t2 - 1]
    # start decoding
    for i in range(SH):
        if s2[i] == 1:  # 第一个操作
            ON = len(MA[mm[i]].Op)
            t = time[f_index][s1[i]][s2[i] - 1][mm[i]]  # 模糊加工时间 [a, b, c]
            if ON > 0:
                Index1 = -1
                for j in range(ON):
                    # 使用 fuzzy_rank 比较间隙时间和加工时间
                    gap_rank = fuzzy_rank(MA[mm[i]].GapT[j])
                    t_rank = fuzzy_rank(t)
                    if gap_rank > t_rank:  # 遍历机器 mm[i] 的模糊空闲时间列表 GapT，寻找可以插入当前操作的位置 Index1
                        Index1 = j
                        break
                if Index1 != -1:  # 插入操作
                    Index1 = MA[mm[i]].Op[Index1]
                    tmp = s1[i]
                    for j in range(i, Index1, -1):
                        s1[j] = s1[j - 1]
                    s1[Index1] = tmp
                    tmp = s2[i]
                    for j in range(i, Index1, -1):
                        s2[j] = s2[j - 1]
                    s2[Index1] = tmp
                    tmp = mm[i]
                    for j in range(i, Index1, -1):
                        mm[j] = mm[j - 1]
                    mm[Index1] = tmp
                    # 将操作序列、操作计数和机器选择数组中的元素向后移动，为插入操作腾出位置
                    for j in range(ON):
                        if MA[mm[Index1]].Op[j] >= Index1:
                            MA[mm[Index1]].Op[j] = MA[mm[Index1]].Op[j] + 1
                    for k in range(TM):
                        if k != mm[Index1]:
                            ON2 = len(MA[k].Op)
                            for h in range(ON2):
                                if MA[k].Op[h] > Index1 and MA[k].Op[h] < i:
                                    MA[k].Op[h] = MA[k].Op[h] + 1
                    # 更新当前机器的操作列表，更新其他机器的操作列表
                    MA[mm[Index1]].Op.append(Index1)
                    MA[mm[Index1]].Op.sort()  # 升序
                    IIndex = find_all_index(MA[mm[Index1]].Op, Index1)  # 找到插入位置的索引
                    # 如果插入位置是列表开头，开始时间为 [0,0,0]；否则，开始时间为前一个操作的模糊完成时间
                    if IIndex[0] == 0:
                        start[s1[Index1]][s2[Index1] - 1] = [0, 0, 0]
                    else:
                        LastOp = MA[mm[Index1]].Op[IIndex[0] - 1]
                        start[s1[Index1]][s2[Index1] - 1] = fuzzy_max([0, 0, 0], finish[s1[LastOp]][s2[LastOp] - 1])  # 使用 fuzzy_max
                    finish[s1[Index1]][s2[Index1] - 1] = fuzzy_add(start[s1[Index1]][s2[Index1] - 1], t)  # 模糊加法
                    ON = len(MA[mm[Index1]].Op)
                    for j in range(ON):
                        if j == 0:
                            MA[mm[Index1]].GapT[j] = start[s1[Index1]][s2[Index1] - 1]
                        else:
                            LastOp = MA[mm[Index1]].Op[j - 1]
                            MA[mm[Index1]].GapT[j] = [start[s1[Index1]][s2[Index1] - 1][0] - finish[s1[LastOp]][s2[LastOp] - 1][0],
                                                      start[s1[Index1]][s2[Index1] - 1][1] - finish[s1[LastOp]][s2[LastOp] - 1][1],
                                                      start[s1[Index1]][s2[Index1] - 1][2] - finish[s1[LastOp]][s2[LastOp] - 1][2]]
                        MA[mm[Index1]].MFT[j] = finish[s1[Index1]][s2[Index1] - 1]
                    mt[mm[Index1]] = MA[mm[Index1]].MFT[ON - 1]
                else:  # Index1 == -1
                    start[s1[i]][s2[i] - 1] = fuzzy_max(MA[mm[i]].MFT[ON - 1], finish[s1[i]][s2[i]])  # 使用 fuzzy_max
                    mt[mm[i]] = fuzzy_add(start[s1[i]][s2[i] - 1], t)  # 模糊加法
                    finish[s1[i]][s2[i] - 1] = mt[mm[i]]
                    MA[mm[i]].Op.append(i)
                    gap = [start[s1[i]][s2[i] - 1][0] - MA[mm[i]].MFT[ON - 1][0],
                           start[s1[i]][s2[i] - 1][1] - MA[mm[i]].MFT[ON - 1][1],
                           start[s1[i]][s2[i] - 1][2] - MA[mm[i]].MFT[ON - 1][2]]
                    MA[mm[i]].GapT.append(gap)
                    MA[mm[i]].MFT.append(mt[mm[i]])
            else:  # ON == 0
                mt[mm[i]] = fuzzy_add(finish[s1[i]][s2[i]], t)  # 模糊加法
                start[s1[i]][s2[i] - 1] = finish[s1[i]][s2[i]]
                finish[s1[i]][s2[i] - 1] = mt[mm[i]]
                MA[mm[i]].Op.append(i)
                MA[mm[i]].GapT.append(start[s1[i]][s2[i] - 1])
                MA[mm[i]].MFT.append(mt[mm[i]])
    newp = s1
    return newp

def AS2FAS(p_chrom, m_chrom, FJ, N, H, TM, time, f_index):  # 从后遍历
    SH = len(p_chrom)
    opmax = int(max(H))
    # 初始化，finish和start矩阵记录每个操作的模糊开始与完成时间
    finish = np.zeros(shape=(N, opmax, 3))
    start = np.zeros(shape=(N, opmax, 3))
    mt = np.zeros((TM, 3))  # 记录每个机器的当前模糊时间 [a, b, c]
    MA = []  # 是机器对象列表，用于跟踪每个机器上的操作、模糊空闲时间和模糊完成时间
    for i in range(TM):  # creat a object array
        MA.append(Machine())

    s1 = p_chrom  # 操作序列
    s2 = np.zeros(SH, dtype=int)  # 操作队列
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    mm = np.zeros(SH, dtype=int)  # 机器分配
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t4 = 0
        for k in range(t1):  # sum from 0 to t1-1
            t4 = t4 + H[k]
        mm[i] = m_chrom[t4 + t2 - 1]
    # start decoding from end
    for i in range(SH - 1, -1, -1):
        if s2[i] == H[s1[i]]:  # 最后一个操作
            ON = len(MA[mm[i]].Op)
            t = time[f_index][s1[i]][s2[i] - 1][mm[i]]  # 模糊加工时间 [a, b, c]
            if ON > 0:
                Index1 = -1
                for j in range(ON):
                    # 使用 fuzzy_rank 比较间隙时间和加工时间
                    gap_rank = fuzzy_rank(MA[mm[i]].GapT[j])
                    t_rank = fuzzy_rank(t)
                    if gap_rank > t_rank:  # 遍历机器 mm[i] 的模糊空闲时间列表 GapT，寻找可以插入当前操作的位置 Index1
                        Index1 = j
                        break
                if Index1 != -1:  # 插入操作
                    Index1 = MA[mm[i]].Op[Index1]
                    tmp = s1[i]
                    for j in range(i, Index1):
                        s1[j] = s1[j + 1]
                    s1[Index1] = tmp
                    tmp = s2[i]
                    for j in range(i, Index1):
                        s2[j] = s2[j + 1]
                    s2[Index1] = tmp
                    tmp = mm[i]
                    for j in range(i, Index1):
                        mm[j] = mm[j + 1]
                    mm[Index1] = tmp
                    # 将操作序列、操作计数和机器选择数组中的元素向前移动，为插入操作腾出位置
                    for j in range(ON):
                        if MA[mm[Index1]].Op[j] <= Index1:
                            MA[mm[Index1]].Op[j] = MA[mm[Index1]].Op[j] - 1
                    for k in range(TM):
                        if k != mm[Index1]:
                            ON2 = len(MA[k].Op)
                            for h in range(ON2):
                                if MA[k].Op[h] < Index1 and MA[k].Op[h] > i:
                                    MA[k].Op[h] = MA[k].Op[h] - 1
                    # 更新当前机器的操作列表，更新其他机器的操作列表
                    MA[mm[Index1]].Op.append(Index1)
                    MA[mm[Index1]].Op.sort(reverse=True)  # 降序
                    IIndex = find_all_index(MA[mm[Index1]].Op, Index1)
                    if IIndex[0] == 0:
                        start[s1[Index1]][s2[Index1] - 1] = fuzzy_max([0, 0, 0], finish[s1[Index1]][s2[Index1]])  # 使用 fuzzy_max
                    else:
                        LastOp = MA[mm[Index1]].Op[IIndex[0] - 1]
                        start[s1[Index1]][s2[Index1] - 1] = fuzzy_max(finish[s1[Index1]][s2[Index1]], finish[s1[LastOp]][s2[LastOp] - 1])  # 使用 fuzzy_max
                    finish[s1[Index1]][s2[Index1] - 1] = fuzzy_add(start[s1[Index1]][s2[Index1] - 1], t)  # 模糊加法
                    ON = len(MA[mm[Index1]].Op)
                    for j in range(ON):
                        Index1 = MA[mm[Index1]].Op[j]
                        if j == 0:
                            MA[mm[Index1]].GapT[j] = start[s1[Index1]][s2[Index1] - 1]
                        else:
                            LastOp = MA[mm[Index1]].Op[j - 1]
                            MA[mm[Index1]].GapT[j] = [start[s1[Index1]][s2[Index1] - 1][0] - finish[s1[LastOp]][s2[LastOp] - 1][0],
                                                      start[s1[Index1]][s2[Index1] - 1][1] - finish[s1[LastOp]][s2[LastOp] - 1][1],
                                                      start[s1[Index1]][s2[Index1] - 1][2] - finish[s1[LastOp]][s2[LastOp] - 1][2]]
                        MA[mm[Index1]].MFT[j] = finish[s1[Index1]][s2[Index1] - 1]
                    mt[mm[Index1]] = MA[mm[Index1]].MFT[0]  # 取降序后的第一个
                else:  # Index1 == -1
                    start[s1[i]][s2[i] - 1] = fuzzy_max(MA[mm[i]].MFT[0], finish[s1[i]][s2[i]])  # 使用 fuzzy_max
                    mt[mm[i]] = fuzzy_add(start[s1[i]][s2[i] - 1], t)  # 模糊加法
                    finish[s1[i]][s2[i] - 1] = mt[mm[i]]
                    MA[mm[i]].Op.append(i)
                    gap = [start[s1[i]][s2[i] - 1][0] - MA[mm[i]].MFT[0][0],
                           start[s1[i]][s2[i] - 1][1] - MA[mm[i]].MFT[0][1],
                           start[s1[i]][s2[i] - 1][2] - MA[mm[i]].MFT[0][2]]
                    MA[mm[i]].GapT.append(gap)
                    MA[mm[i]].MFT.append(mt[mm[i]])
            else:  # ON == 0
                mt[mm[i]] = fuzzy_add(finish[s1[i]][s2[i]], t)  # 模糊加法
                start[s1[i]][s2[i] - 1] = finish[s1[i]][s2[i]]
                finish[s1[i]][s2[i] - 1] = mt[mm[i]]
                MA[mm[i]].Op.append(i)
                MA[mm[i]].GapT.append(start[s1[i]][s2[i] - 1])
                MA[mm[i]].MFT.append(mt[mm[i]])
    newp = s1
    return newp

def EnergysavingDHFJSP(p_chrom, m_chrom, f_chrom, fitness, N, H, TM, time, SH, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    IP = []
    FJ = []
    for f in range(F):
        P.append([])
        IP.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        IP[t3].append(i)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])
    for f in range(F):
        # 运用 SAS2AS 和 AS2FAS
        P[f] = SAS2AS(P[f], m_chrom, FJ[f], N, H, TM, time, f)  # complete test
        P[f] = AS2FAS(P[f], m_chrom, FJ[f], N, H, TM, time, f)
    newf = f_chrom
    newm = m_chrom
    newp = np.zeros(SH, dtype=int)
    for f in range(F):
        L = len(IP[f])
        for i in range(L):
            newp[IP[f][i]] = P[f][i]
    return newp, newm, newf

def fuzzy_add(z, t):
    return (z[0] + t[0], z[1] + t[1], z[2] + t[2])

def fuzzy_rank(z):
    return (z[0] + 2 * z[1] + z[2]) / 4

def fuzzy_max(z, t):
    if fuzzy_rank(z) > fuzzy_rank(t):
        return z if z[0] > t[0] and z[1] > t[1] and z[2] > t[2] else t
    elif fuzzy_rank(z) == fuzzy_rank(t):
        return z if z[1] > t[1] else t
    return t

class individual:
    def __init__(self):
        self.n = 0  # the number of being dominated of individual i
        self.p = []  # the list of index which dominate individual i

# each pareto front
class front:
    def __init__(self):
        self.f = []  # the list of ith front

def FastNDS(fitness, ps):
    # store the indexs of top rank ps solutions
    TopRank = []
    L = np.size(fitness, 0)
    # create a object array
    S = []  # solutions
    for i in range(L):
        S.append(individual())
    # create pareto front
    F = []
    rank = 0
    F.append(front())
    # find first front
    for i in range(L):
        for j in range(L):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(2):
                if (fitness[i][k] > fitness[j][k]):
                    dom_more = dom_more + 1
                elif (fitness[i][k] == fitness[j][k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1

            if dom_less == 0 and dom_equal != 2:  # i is dominated by j
                S[i].n = S[i].n + 1
            elif dom_more == 0 and dom_equal != 2:  # i dominate j
                S[i].p.append(j)
        if S[i].n == 0:  # add i into pareto front
            F[rank].f.append(i)
    # find the other front
    while len(F[rank].f) > 0:
        Q = []
        fL = len(F[rank].f)
        for i in range(fL):
            x = F[rank].f[i]
            if len(S[x].p) > 0:
                pL = len(S[x].p)
                for j in range(pL):
                    k = S[x].p[j]
                    S[k].n = S[k].n - 1
                    if S[k].n == 0:
                        Q.append(k)
        rank = rank + 1
        F.append(front())  # create a new front
        F[rank].f = Q

    # crowding distance strategy
    fL = len(F); fL = fL - 1  # the last front is empty
    obj = fitness
    currentindex = 0
    for Frt in range(fL):
        ffL = len(F[Frt].f)
        y = np.zeros(shape=(ffL, 2))
        for i in range(ffL):
            t = F[Frt].f[i]
            y[i, :] = obj[t, :]
        crowd = []
        crowd = np.zeros(shape=(ffL, 2))
        for i in range(2):
            y0 = y[:, i]
            sort_index = np.argsort(y0)  # ascending sort
            fmin = y[sort_index[0]][i]
            fmax = y[sort_index[ffL - 1]][i]
            if ffL == 1 or fmax == fmin:
                for j in range(ffL):
                    crowd[j][i] = 1
            else:
                sort_obj = []
                for j in range(ffL):
                    sort_obj.append(y[sort_index[j]][i])
                gap = fmax - fmin
                for j in range(ffL):
                    if j == 0:
                        crowd[sort_index[j]][i] = (sort_obj[1] - sort_obj[0]) / gap
                    elif j == ffL - 1:
                        crowd[sort_index[j]][i] = (sort_obj[j] - sort_obj[j - 1]) / gap
                    else:
                        crowd[sort_index[j]][i] = (sort_obj[j + 1] - sort_obj[j - 1]) / gap
        crowd[:, 0] = crowd[:, 0] + crowd[:, 1]
        tmp = []; sort_index = []
        sort_index = np.argsort(-crowd[:, 0])  # add '-' means sort in descending
        # tmp = F[Frt].f  # light copy
        l = len(sort_index)
        for i in range(l):  # deep copy
            tmp.append(F[Frt].f[i])
        for i in range(l):
            F[Frt].f[i] = tmp[sort_index[i]]

    count = 0
    for Frt in range(fL):
        ffL = len(F[Frt].f)
        for i in range(ffL):
            TopRank.append(F[Frt].f[i])
            count = count + 1
            if count == ps:
                break
        if count == ps:
            break
    return TopRank








Combination=[[10,2],[20,2],[30,2],[40,2],\
             [20,3],[30,3],[40,3],[50,3],\
             [40,4],[50,4],[100,4],\
             [50,5],[100,5],[150,5],\
             [100,6],[150,6],[200,6],\
             [100,7],[150,7],[200,7],\
             [0,0]]
datapath='E:/bishe/DHFJS/DQCE-code/DQCE-code/DQCE-code/fuzzy/'
FileName=[];ResultPath=[]
for i in range(20):
    J=Combination[i][0]
    F1=Combination[i][1]
    O=5
    temp = datapath +  str(J) +'J' + str(F1)+ 'F_fuzzy' + '.txt'
    temp2 = str(J) +'J' + str(F1)+ 'F'
    FileName.append(temp);
    ResultPath.append(temp2)
FileName.append(datapath + 'realworld.txt')
ResultPath.append('realworld')
TF=21


FileName=np.array(FileName);FileName.reshape(TF,1)
ResultPath=np.array(ResultPath);ResultPath.reshape(TF,1)
#read the parameter of algorithm such as popsize, crossover rate, mutation rate
f= open("parameter.txt", "r", encoding='utf-8')
ps,Pc,Pm,lr,batch_size,EPSILON,GAMMA,MEMORY_CAPACITY = f.read().split(' ')
#100 1.0 0.2 0.001 16 0.9 0.9 512
#种群大小 交叉概率 变异概率 学习率（步长） 批处理大小 探索率 折扣因子 经验池大小
ps=int(ps);Pc=float(Pc);Pm=float(Pm);lr=float(lr);batch_size=int(batch_size)
EPSILON=float(EPSILON);GAMMA=float(GAMMA);MEMORY_CAPACITY=int(MEMORY_CAPACITY)
IndependentRun=10
#独立运行10次

lr=0.001;batch_size=16;
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 512

TARGET_REPLACE_ITER = 7   # 目标网络更新频率=7
N_ACTIONS = 9  # 9种可选算子
EPOCH=1 #训练轮次=1


print(torch.cuda.is_available())


# # 遍历FileName列表并打印每个文件路径
# for index, file_path in enumerate(FileName):
#     print(f"Index {index}: {file_path}")

#更改CCF值，即可选择不同的文件读取
# Index 0: 10J2F.txt
# Index 1: 20J2F.txt
# Index 2: 30J2F.txt
# Index 3: 40J2F.txt
# Index 4: 20J3F.txt
# Index 5: 30J3F.txt
# Index 6: 40J3F.txt
# Index 7: 50J3F.txt
# Index 8: 40J4F.txt
# Index 9: 50J4F.txt
# Index 10: 100J4F.txt
# Index 11: 50J5F.txt
# Index 12: 100J5F.txt
# Index 13: 150J5F.txt
# Index 14: 100J6F.txt
# Index 15: 150J6F.txt
# Index 16: 200J6F.txt
# Index 17: 100J7F.txt
# Index 18: 150J7F.txt
# Index 19: 200J7F.txt
CCF=0


#10j2f
#10作业数量 2工厂数量 5每个作业的工序数
# 1 1 5 第一个工厂第一个作业 5道工序
# 1工序号 5可用机器数 1 5机器1（时间5） 2 18机器2（时间18） 3 12机器3（时间12） 4 18机器4（时间18） 5 15机器5（时间15）
# 2 3 1 18 2 17 3 11
# 3 3 1 16 4 17 5 12
# 4 4 1 19 2 16 3 8 4 11
# 5 2 1 12 4 19

#十个工件 每个工件需要五步 每个工件五步只能在一个工厂 每步在一个工厂中的不同的机器上时间不同


for file in range(CCF, CCF + 1):
    N, F, TM, H, SH, NM, M, time, ProF = DataReadDHFJSP(FileName[file])

    print(f"N={N}, F={F}, TM={TM}, H={H}, SH={SH}, time shape={time.shape}")
    print(f"time[0][0][0]={time[0][0][0]}")  # 打印第一个工序的模糊时间

    MaxNFEs = 200 * SH  # 最大函数评估次数（停止条件）
    # create filepath to store the pareto solutions set for each independent run
    respath = 'DQNV9+ES\\'
    sprit = '\\'  # 路径分隔符（Windows系统）
    respath = respath + ResultPath[file]
    isExist = os.path.exists(respath)
    # if the result path has not been created
    if not isExist:
        currentpath = os.getcwd()
        os.makedirs(currentpath + sprit + respath)
    print(ResultPath[file], 'is being Optimizing\n')
    # start independent run for GMA
    for rround in range(1):
        p_chrom, m_chrom, f_chrom = initial(N, H, SH, NM, M, ps, F)  # 初始化种群
        # p_chrom：作业工序顺序（形状(ps, SH)）。
        # m_chrom：机器分配方案（形状(ps, SH)）。
        # f_chrom：工厂分配方案（形状(ps, N)）。
        fitness = np.zeros(shape=(ps, 3))

        NFEs = 0  # number of function evaluation
        # calucate fitness of each solution
        for i in range(ps):  # 计算初始化种群的适应度（完成时间与能量消耗）
            fitness[i, 0], fitness[i, 1], fitness[i, 2] = CalfitDHFJFP(p_chrom[i, :], m_chrom[i, :], f_chrom[i, :], N, H, SH, F, TM, time)
        # 存储最佳的选择
        AP = []; AM = []; AF = []; AFit = []  # Elite archive # 操作序列、机器选择、工厂分配、适应度
        i = 1
        # 创建模型
        N_STATES = 2 * SH + N + 3  # 状态空间=操作序列+机器选择+工厂分配
        CountOpers = np.zeros(N_ACTIONS)  # 用于记录每种操作被执行的次数
        PopCountOpers = []  # 用于存储种群中每个个体的操作计数
        dq_net = DQN(N_STATES, N_ACTIONS, BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA=GAMMA, \
                     MEMORY_CAPACITY=MEMORY_CAPACITY, TARGET_REPLACE_ITER=TARGET_REPLACE_ITER)
        Loss = []
        while NFEs < MaxNFEs:
            print(FileName[file] + ' round ', rround + 1, 'iter ', i)
            i = i + 1
            ChildP = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildM = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildF = np.zeros(shape=(2 * ps, N), dtype=int)
            ChildFit = np.zeros(shape=(2 * ps, 3))
            # 使用锦标赛选择方法选择父代个体
            P_pool, M_pool, F_pool = tournamentSelection(p_chrom, m_chrom, f_chrom, fitness, ps, SH, N)
            # 生成子代
            for j in range(ps):
                Fit1 = np.zeros(3);
                Fit2 = np.zeros(3)
                P1, M1, F1, P2, M2, F2 = evolution(P_pool, M_pool, F_pool, j, Pc, Pm, ps, SH, N, H, NM, M)
                # 交叉和变异生成子代个体
                Fit1[0], Fit1[1], Fit1[2] = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)
                Fit2[0], Fit2[1], Fit2[2] = CalfitDHFJFP(P2, M2, F2, N, H, SH, F, TM, time)
                # 计算子代适应度
                NFEs = NFEs + 2;
                t1 = j * 2;
                t2 = j * 2 + 1
                ChildP[t1, :] = copy.copy(P1); ChildM[t1, :] = copy.copy(M1); ChildF[t1, :] = copy.copy(F1); ChildFit[t1, :] = Fit1
                ChildP[t2, :] = copy.copy(P2); ChildM[t2, :] = copy.copy(M2); ChildF[t2, :] = copy.copy(F2); ChildFit[t2, :] = Fit2
            QP = np.vstack((p_chrom, ChildP))
            QM = np.vstack((m_chrom, ChildM))
            QF = np.vstack((f_chrom, ChildF))
            QFit = np.vstack((fitness, ChildFit))
            # 合并子代与父代，vstake竖直堆叠数组
            QP, QM, QF, QFit = DeleteReapt(QP, QM, QF, QFit, ps)
            RQFit = QFit[:, 0:2]

            TopRank = FastNDS(RQFit, ps)
            p_chrom = QP[TopRank, :];
            m_chrom = QM[TopRank, :];
            f_chrom = QF[TopRank, :];
            fitness = QFit[TopRank, :]
            # 快速非支配解排序

            PF = pareto(fitness)
            if len(AFit) == 0:
                AP = copy.copy(p_chrom[PF, :])
                AM = copy.copy(m_chrom[PF, :])
                AF = copy.copy(f_chrom[PF, :])
                AFit = copy.copy(fitness[PF, :])
            # 非支配解存入精英存档

            # Elite strategy
            PF = pareto(fitness)
            if len(AFit) == 0:
                AP = p_chrom[PF, :]
                AM = m_chrom[PF, :]
                AF = f_chrom[PF, :]
                AFit = fitness[PF, :]
            else:
                AP = np.vstack((AP, p_chrom[PF, :]))
                AM = np.vstack((AM, m_chrom[PF, :]))
                AF = np.vstack((AF, f_chrom[PF, :]))
                AFit = np.vstack((AFit, fitness[PF, :]))
            PF = pareto(AFit)
            AP = AP[PF, :];
            AM = AM[PF, :];
            AF = AF[PF, :];
            AFit = AFit[PF, :];
            AP, AM, AF, AFit = DeleteReaptE(AP, AM, AF, AFit)

            # 在精英存档中使用局部搜索策略
            L = len(AFit)
            current_state = np.zeros(N_STATES, dtype=int)
            next_state = np.zeros(N_STATES, dtype=int)
            # 使用 DQN 选择局部搜索动作
            for l in range(L):
                current_state[0:SH] = copy.copy(AP[l, :])
                current_state[SH:SH * 2] = copy.copy(AM[l, :])
                current_state[SH * 2:N_STATES - 3] = copy.copy(AF[l, :])  # 保留最后 3 位给 eta
                current_state[N_STATES - 3:] = copy.copy(AFit[l, :])  # 存储 eta 值

                action = dq_net.choose_action(current_state)
                k = int(action)
                if k == 0:
                    P1, M1, F1 = N6(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 1:
                    P1, M1, F1 = SwapOF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time)
                elif k == 2:
                    P1, M1, F1 = RandFA(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 3:
                    P1, M1, F1 = RandMS(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)
                elif k == 4:
                    P1, M1, F1 = InsertOF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time)
                elif k == 5:
                    P1, M1, F1 = InsertIF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, F)
                elif k == 6:
                    P1, M1, F1 = SwapIF(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, F)
                elif k == 7:
                    P1, M1, F1 = RankFA(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F, ProF)
                elif k == 8:
                    P1, M1, F1 = RankMS(AP[l, :], AM[l, :], AF[l, :], AFit[l, :], N, H, SH, time, TM, NM, M, F)

                Fit1 = np.zeros(3)  # 存储基于模糊数 eta 值
                Fit1[0], Fit1[1], Fit1[2] = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)  # 计算基于模糊数 eta
                NFEs = NFEs + 1
                dom = NDS(Fit1, AFit[l, :])
                if dom == 1:  # 生成非支配解奖励为1
                    AP[l, :] = copy.copy(P1)
                    AM[l, :] = copy.copy(M1)
                    AF[l, :] = copy.copy(F1)
                    AFit[l, :] = copy.copy(Fit1)
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    reward = 5
                elif dom == 0 and AFit[l][0] != Fit1[0] and AFit[l][1] != Fit1[1]:  # 生成支配解奖励为10
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    reward = 10
                else:
                    reward = 0
                next_state[0:SH] = copy.copy(P1)
                next_state[SH:SH * 2] = copy.copy(M1)
                next_state[SH * 2:N_STATES - 3] = copy.copy(F1)
                next_state[N_STATES - 3:] = copy.copy(Fit1)  # 存储 eta 值
                dq_net.store_transition(current_state, action, reward, next_state)
                if dq_net.memory_counter > 50:  # 经验池中数据大于 50 才开始训练
                    for epoch in range(EPOCH):
                        loss = dq_net.learn()  # 调用 DQN 的 learn 方法训练一次
                        Loss.append(loss)

            # 节能策略
            L = len(AFit)
            for j in range(L):
                P1, M1, F1 = EnergysavingDHFJSP(AP[j, :], AM[j, :], AF[j, :], AFit[j, :], N, H, TM, time, SH, F)
                Fit1 = np.zeros(3)  # 存储基于模糊数 eta 值
                Fit1[0], Fit1[1], Fit1[2] = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)  # 计算基于模糊数 eta
                NFEs = NFEs + 1
                if NDS(Fit1, AFit[j, :]) == 1:  # 假设 NDS 处理模糊数或 eta 标量
                    AP[j, :] = copy.copy(P1)
                    AM[j, :] = copy.copy(M1)
                    AF[j, :] = copy.copy(F1)
                    AFit[j, :] = copy.copy(Fit1)
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                elif NDS(Fit1, AFit[j, :]) == 0:
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))

        # 展示FA/Ms/OA
        PF = pareto(AFit)
        AP = AP[PF, :]
        AM = AM[PF, :]
        AF = AF[PF, :]
        AFit = AFit[PF, :]

        # 再次提取帕累托前沿以确保是最新的
        PF = pareto(AFit)
        l = len(PF)
        obj = AFit[:, 0:2]
        newobj = []
        unique_indices = []

        for i in range(l):
            newobj.append(obj[PF[i], :])
            unique_indices.append(PF[i])

        # 去重
        newobj = np.array(newobj)
        unique_indices = np.array(unique_indices)
        _, unique_idx = np.unique(newobj, axis=0, return_index=True)
        unique_indices = unique_indices[unique_idx]

        # 在终端中显示每个帕累托前沿解的操作序列、机器选择和工厂分配
        for i, idx in enumerate(unique_indices):
            print(f"解 {i + 1}:")
            print(f"操作序列: {AP[idx]}")
            print(f"机器选择: {AM[idx]}")
            print(f"工厂分配: {AF[idx]}")
            print(f"适应度: {AFit[idx]}")
            print()

        # 保存帕累托前沿的目标值到文件
        tmp = 'res'
        resPATH = respath + sprit + tmp + str(rround + 1) + '.txt'
        with open(resPATH, "w", encoding='utf-8') as f:
            for i in range(len(newobj)):
                item = f"{newobj[i][0]:5.2f} {newobj[i][1]:6.2f}\n"
                f.write(item)

        # 保存损失值到文件
        with open('loss.txt', "w", encoding='utf-8') as f:
            for loss in Loss:
                f.write(f"{loss}\n")

        print("帕累托前沿解的目标值已保存到文件。")

        for i, idx in enumerate(unique_indices):
            print(f"Solution {i + 1}:")
            print(f"Operation Sequence: {AP[idx]}")
            print(f"Machine Selection: {AM[idx]}")
            print(f"Factory Assignment: {AF[idx]}")
            print(f"Fitness: {AFit[idx]}")
            print()

            # 绘制甘特图
            plt.figure(figsize=(12, 8))
            machine_info = {}  # 用于存储机器信息
            machine_start_time = {}  # 用于存储每个机器的当前开始时间 [a, b, c]
            colors = plt.cm.tab20(np.linspace(0, 1, N))  # 按任务分配颜色

            for job in range(N):
                start_time = [0, 0, 0]  # 初始化模糊开始时间
                for op in range(H[job]):
                    # 获取操作信息
                    factory = AF[idx][job]
                    machine = AM[idx][job * H[job] + op]
                    t = time[int(factory)][job][op][int(machine)]  # 模糊时间 [a, b, c]

                    # 机器标签格式为 FfMm
                    machine_label = f"F{factory + 1}M{machine + 1}"

                    # 如果机器标签不在字典中，则初始化
                    if machine_label not in machine_info:
                        machine_info[machine_label] = []
                        machine_start_time[machine_label] = [0, 0, 0]

                    # 计算模糊开始时间 (使用前一个任务的完成时间)
                    if op > 0:
                        prev_op_finish = machine_info[machine_label][-1]['finish']
                        start_time = [max(start_time[0], prev_op_finish[0]),
                                      max(start_time[1], prev_op_finish[1]),
                                      max(start_time[2], prev_op_finish[2])]
                    # 计算模糊完成时间
                    finish_time = [start_time[0] + t[0], start_time[1] + t[1], start_time[2] + t[2]]

                    # 记录任务信息
                    task_info = {
                        'job': job + 1,
                        'op': op + 1,
                        'start': start_time,  # 模糊开始时间 [a, b, c]
                        'duration': t,  # 模糊持续时间 [a, b, c]
                        'finish': finish_time  # 模糊完成时间 [a, b, c]
                    }
                    machine_info[machine_label].append(task_info)

                    # 更新机器的模糊开始时间为当前任务的完成时间
                    machine_start_time[machine_label] = finish_time

            # 按工厂和机器编号排序
            sorted_machine_labels = sorted(machine_info.keys(),
                                           key=lambda x: (int(x.split('F')[1].split('M')[0]), int(x.split('M')[1])))

            # 绘制甘特图
            y_pos = np.arange(len(sorted_machine_labels))  # 每个机器的y坐标位置

            for machine_idx, machine_label in enumerate(sorted_machine_labels):
                tasks = machine_info[machine_label]
                for task in tasks:
                    start = task['start']
                    duration = task['duration']
                    # 计算三角形顶点位置
                    left = (start[0] + start[1] + start[2]) / 3  # 使用平均值作为左顶点位置
                    peak = start[1] + duration[1]  # 顶点在 b 位置
                    right = start[2] + duration[2]  # 右顶点在 c 位置
                    y_base = machine_idx - 0.5
                    y_peak = machine_idx + 0.5

                    # 绘制三角形
                    x = [left, peak, right]
                    y = [y_base, y_peak, y_base]
                    plt.fill(x, y, color=colors[task['job'] - 1], alpha=0.7, edgecolor='black')

                    # 在三角形内部标注操作标签
                    center_x = (left + right) / 2
                    center_y = machine_idx
                    plt.text(center_x, center_y, f'O{task["job"]}.{task["op"]}', ha='center', va='center',
                             color='white', fontsize=10)

                    # 标注三角模糊数 (a, b, c)
                    plt.text(peak, y_peak + 0.2, f'({duration[0]:.1f},{duration[1]:.1f},{duration[2]:.1f})',
                             ha='center', va='bottom', color='black', fontsize=8)

            # 设置图表属性
            plt.yticks(y_pos, sorted_machine_labels)
            plt.xlabel('Time')
            plt.ylabel('Factories-Machines')
            plt.title(f'Solution {i + 1} Gantt Chart')
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.show()

    print('finish ' + FileName[file])
print('finish running')