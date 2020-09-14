import pandas as pd
import numpy as np
import math


class CSV:

    def __init__(self, nomeArquivo):
        self.dados = None
        self.linhas = 0
        self.colunas = 0
        self.readFile(nomeArquivo)
        self.qtdAtributos = self.colunas - 1
        self.qtdClasses = self.numberOfDifferentClasses()
        self.qtdNeuroniosOculta = int(math.sqrt(self.qtdAtributos * self.qtdClasses))

    def readFile(self, nomeArquivo):
        tabela = pd.read_csv(nomeArquivo)

        self.dados = self.toMatrix(tabela.to_numpy(dtype=int))

    def toMatrix(self, arrayList):
        self.linhas = len(arrayList)
        self.colunas = len(arrayList[0])

        matriz = []  # matriz é uma lista de listas
        for i in range(self.linhas):
            linha = []  # cada linha da matriz é uma lista
            for j in range(self.colunas):
                linha.append(arrayList[i][j])
            matriz.append(linha)

        return matriz

    def numberOfDifferentClasses(self):
        qtdd = set()
        for i in range(self.linhas):
            qtdd.add(self.dados[i][self.colunas - 1])

        return len(qtdd)

    def getDados(self):
        return self.dados

    def getQtdAtributos(self):
        return self.qtdAtributos

    def getQtdClasses(self):
        return self.qtdClasses

    def getQtdNeuroniosOculta(self):
        return self.qtdNeuroniosOculta


class RedeNeural:

    def __init__(self, arquivo, funcTransferencia):
        self.arquivo = arquivo
        self.qtdNeuroniosEntrada = arquivo.getQtdAtributos()
        self.qtdNeuroniosSaida = arquivo.getQtdClasses()
        self.qtdNeuroniosOculta = arquivo.getQtdNeuroniosOculta()
        self.funcTransferencia = funcTransferencia
        self.taxaAprendizado = 0.01

        self.acertos = 0
        self.qtdRegistros = 0
        self.pesosEntradaOculta = None
        self.pesosEntradaSaida = None
        self.pesosOcultaSaida = None
        self.errosSaida = None
        self.errosOculta = None
        self.saidaOculta = None
        self.entrada = None
        self.desejado = None
        self.obtido = None

        self.inicializarPesos(-0.5, 0.5)
        self.fit()

    def inicializarPesos(self, valorMin, valorMax):
        self.pesosEntradaOculta = np.random.uniform(valorMin, valorMax,
                                                    size=(self.qtdNeuroniosOculta, self.qtdNeuroniosEntrada))
        self.pesosEntradaSaida = np.random.uniform(valorMin, valorMax,
                                                   size=(self.qtdNeuroniosSaida, self.qtdNeuroniosOculta))
        print()
        print(self.pesosEntradaOculta)
        print()
        print(self.pesosEntradaSaida)
        print()
        
    def fit(self):
        for j in range(200):
            for i in range(len(self.arquivo.getDados())):
                self.inicializarEntrada(self.arquivo.getDados()[i])
                self.inicializarDesejado(self.arquivo.getDados()[i])
                self.gerarSaidasOculta()
                self.gerarObtido()
                self.encontrarErrosDaSaida()
                self.encontrarErrosOculta()
                self.atualizarPesos(self.pesosOcultaSaida, self.errosSaida, self.saidaOculta)
                self.atualizarPesos(self.pesosEntradaOculta, self.errosOculta, self.entrada)
                print(f'Erro da rede é {self.erroRede()}')

    def inicializarEntrada(self, novaEntrada):
        vetor = []
        for i in range(self.qtdNeuroniosEntrada):
            vetor[i] = [novaEntrada[i]]  # vetor coluna

        self.entrada = np.array(vetor)  # criei um vetor coluna com os valores de uma entrada
        print()
        print(self.entrada)
        print()

    def inicializarDesejado(self, novaEntrada):
        classe = novaEntrada[len(novaEntrada) - 1]  # 1, 2, 3, 4, 5

        vetor = []
        for i in range(self.arquivo.getQtdClasses()):
            if self.funcTransferencia == 'Logística':
                vetor[i] = [0]  # entre 0 e 1
            elif self.funcTransferencia == 'Hiperbólica':
                vetor[i] = [-1]  # entre -1 e 1

        vetor[classe - 1] = [1]  # se a classe correta for 4, coloca "1" no índice 3
        self.desejado = np.array(vetor)
        print()
        print(self.desejado)
        print()

    def encontrarErrosDaSaida(self):
        vetor = []
        for i in range(self.arquivo.getQtdClasses()):
            if self.funcTransferencia == 'Logística':
                # (desejado - obtido) * [(obtido) * (1 - obtido)]
                vetor[i] = [(self.desejado[i][0] - self.obtido[i][0]) * (self.obtido[i][0] * (1 - self.obtido[i][0]))]
            elif self.funcTransferencia == 'Hiperbólica':
                # (desejado - obtido) * (1 - obtido²)
                vetor[i] = [(self.desejado[i][0] - self.obtido[i][0]) * (1 - (self.obtido[i][0] * self.obtido[i][0]))]

        self.errosSaida = np.array(vetor)
        print()
        print(self.errosSaida)
        print()

    def encontrarErrosOculta(self):
        self.errosOculta = self.pesosOcultaSaida @ self.errosSaida

        vetor = []
        for i in range(self.qtdNeuroniosOculta):
            if self.funcTransferencia == 'Logística':
                vetor[i] = [self.errosOculta[i][0] * (self.obtido[i][0] * (1 - self.obtido[i][0]))]
            elif self.funcTransferencia == 'Hiperbólica':
                # (desejado - obtido) * (1 - obtido²)
                vetor[i] = [self.errosOculta[i][0] * (1 - (self.obtido[i][0] * self.obtido[i][0]))]

        self.errosOculta = np.array(vetor)
        print()
        print(self.errosOculta)
        print()

    def erroRede(self):
        soma = 0
        for i in range(len(self.errosSaida)):
            soma += math.pow(self.errosSaida[i][0], 2)
        return soma / 2


if __name__ == '__main__':
    arq = CSV('treinamento.csv')
    teste = CSV('teste.csv')
