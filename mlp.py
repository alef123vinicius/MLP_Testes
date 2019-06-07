# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:47:54 2017

@author: alef1

implementando a multi layer paerceptron
"""
import math as m
import random as r
import csv

def f_net(net):
    return (1/(1 + m.exp(-net)))

def derivada_f_net(net):
    return (net * (1 - net))

def arquitetura(tam_entrada, tam_hidden, tam_saida):
    m = []
    matriz_hidden= mat_aleatoria(-1,1,tam_hidden,tam_entrada) # cada linha representa um neuronio da camada escondida
    matriz_saida = mat_aleatoria(-1,1, tam_saida, tam_hidden) # cada linha representa um neuronio da camada de saida
    
    m.append(matriz_hidden)
    m.append(matriz_saida)
    return m
#criando uma matriz de pesos aleatorios    
def mat_aleatoria(vmin = -1,vmax = 1,linhas = 2, colunas = 2):
    matriz = []
    for i in range(linhas):
        l = []
        for x in range(colunas):
            l.append(r.uniform(vmin,vmax))
        matriz.append(l)
    return matriz
#calculando os net e f net de todas a camadas começando pela oculta
def forward(matriz_pesos, entrada, tam_entrada=2,tam_hidden=2, tam_saida=1):
    net_hidden = []
    f_net_hidden = []
    net_out = []
    f_net_out = []
    for j in range(tam_hidden):
        net_hidden.append(somatorio(entrada,matriz_pesos[0][j]))
        f_net_hidden.append(f_net(net_hidden[j]))
    for o in range((tam_saida)):
        net_out.append(somatorio(f_net_hidden,matriz_pesos[1][o]))
        f_net_out.append(f_net(net_out[o]))
    respostas = []
    respostas.append(net_hidden)
    respostas.append(f_net_hidden)
    respostas.append(net_out)
    respostas.append(f_net_out)
    return respostas

def somatorio(entradas, pesos):
    soma = 0
    for i in range(len(pesos)):
        soma += entradas[i]*pesos[i]
    return soma

def delta_saida(erro, fnet_out, tam_saida):
    delta = []
    for o in range(tam_saida):
        delta.append(erro * derivada_f_net(fnet_out[o]))
    return delta

def delta_hidden(fnet_hidden, wokj, delta_o, tam_hidden):
    delta = []
    for h in range(tam_hidden):
        delta.append(derivada_f_net(fnet_hidden[h]) * (delta_o[h]))
    return delta
#o codigo se repete ate que o erro global seja menor que o threshold            
def backpropagation(matriz_pesos, entrada, saida, tam_dados, tam_hidden, tam_saida, tam_entrada, eta = 0.5, threshold = 0.01):
    erros_totais = 2 * threshold
    z = 1
    while(erros_totais > threshold):
        erros_totais = 0
        for i in range(tam_dados):
            vetor_entrada = entrada[i]
            matriz_respostas = forward(matriz_pesos, vetor_entrada, tam_entrada,tam_hidden, tam_saida) 
            erro = [] 
            for e in range(tam_saida):               
                erro.append(saida[i][e]-matriz_respostas[3][e])
           # print("esperado = ",saida[i],"\n obtido = ",matriz_respostas[3], "\n erro = ",erro)
            somat_erro = 0
            for s in range(tam_saida):
                somat_erro += erro[s]*erro[s]/2
            erros_totais += somat_erro
            delta_o = []
            for d in range(tam_saida):
                delta_o.append(erro[d] * derivada_f_net(matriz_respostas[3][d]))
            delta_h = []
            pesos_out = matriz_pesos[1]
            for k in range(tam_hidden):
                result = 0
                for n in range(tam_saida):
                    result += delta_o[n] *pesos_out[n][k]
                delta_h.append(derivada_f_net(matriz_respostas[1][k]) * result )
            for k in range(tam_saida):
                for n in range(tam_hidden):
                    pesos_out[k][n] += eta * delta_o[k] * matriz_respostas[1][n]
            matriz_pesos[1] = pesos_out
            
            pesos_hidden = matriz_pesos[0]    
            for l in range(tam_hidden):
                for j in range(tam_entrada):
                    pesos_hidden[l][j] += eta * delta_h[l] * vetor_entrada[j]
            matriz_pesos[0] = pesos_hidden
        avg = erros_totais 
        print ("erro: ", avg)
        print("interações = ", z)
        z = z+1
        
def arredonda(valor):
    inteiro = int(m.log2(valor))
    frac = m.log2(valor) - inteiro
    if(frac >= 0.5):
        return inteiro + 1
    return inteiro
    
def main():
     tam_entrada = 3
     tam_hidden = arredonda(tam_entrada)
     tam_saida = 1
     #lendo os valores de entrada
     with open('bluetooth.csv', 'rt') as bluetooth:
         reader = csv.reader(bluetooth, delimiter=':', quoting=csv.QUOTE_NONE)
         dados = []
         for linha in reader:
             dados.append(linha[0].split(","))
     entrada = []
     vet_div = [-92,-87,-91]
     for i in range(len(dados)):
         linha = []
         for j in range(len(dados[i])):
             linha.append(float(dados[i][j]))
         for j in range(len(dados[i])-1):
             linha[j] = linha[j]/vet_div[j]
         entrada.append(linha)
     saida = []
     for i in range(len(entrada)):
         vet_saida = []
         for j in range(tam_saida):
             vet_saida.append(entrada[i][tam_entrada-1-j])
         saida.append(vet_saida)
     tam_dados = len(entrada)
     matriz_pesos = arquitetura(tam_entrada,tam_hidden,tam_saida) 
     backpropagation(matriz_pesos, entrada, saida, tam_dados, tam_hidden, tam_saida, tam_entrada, eta = 0.5, threshold = 0.1)
    
if __name__ == "__main__":
    main()
   