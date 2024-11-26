import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        chave = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > chave:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = chave
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        indice_minimo = i
        for j in range(i + 1, n):
            if arr[j] < arr[indice_minimo]:
                indice_minimo = j
        arr[i], arr[indice_minimo] = arr[indice_minimo], arr[i]
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    meio = len(arr) // 2
    esquerda = arr[:meio]
    direita = arr[meio:]
    
    esquerda = merge_sort(esquerda)
    direita = merge_sort(direita)
    
    return merge(esquerda, direita)

def merge(esquerda, direita):
    resultado = []
    i = j = 0
    
    while i < len(esquerda) and j < len(direita):
        if esquerda[i] <= direita[j]:
            resultado.append(esquerda[i])
            i += 1
        else:
            resultado.append(direita[j])
            j += 1
    
    resultado.extend(esquerda[i:])
    resultado.extend(direita[j:])
    return resultado

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    def particionar(baixo, alto):
        pivo = arr[alto]
        i = baixo - 1
        
        for j in range(baixo, alto):
            if arr[j] <= pivo:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[alto] = arr[alto], arr[i + 1]
        return i + 1
    
    def quick_sort_auxiliar(baixo, alto):
        if baixo < alto:
            pi = particionar(baixo, alto)
            quick_sort_auxiliar(baixo, pi - 1)
            quick_sort_auxiliar(pi + 1, alto)
    
    quick_sort_auxiliar(0, len(arr) - 1)
    return arr

def salvar_saida_ordenada(arr, nome_arquivo):
    with open(nome_arquivo, 'w') as arquivo_saida:
        for elemento in arr:
            arquivo_saida.write(str(elemento) + '\n')

def medir_tempo(algoritmo, arr):
    inicio = timeit.default_timer()
    array_ordenado = algoritmo(arr.copy())
    fim = timeit.default_timer()
    tempo_execucao = fim - inicio
    return tempo_execucao, array_ordenado

def funcao_ajuste_quadratica(x, a, b, c):
    return a * x**2 + b * x + c

def funcao_ajuste_logaritmica(x, a, b):
    return a * x * np.log2(x) + b

def comparar_tempos(tempos_reais, tempos_medios, tamanhos, nome_algoritmo, funcao_ajuste=None):
    menor_tempo = min(tempos_reais)
    maior_tempo = max(tempos_reais)
    media_tempo = np.mean(tempos_reais)

    plt.scatter(tamanhos, tempos_reais, label=f'Tempo Real\nMédia: {media_tempo:.4f}\nMínimo: {menor_tempo:.4f}\nMáximo: {maior_tempo:.4f}', s=20)

    if funcao_ajuste:
        parametros_ajustados, _ = curve_fit(funcao_ajuste, tamanhos, tempos_reais)
        curva_ajustada = funcao_ajuste(np.array(tamanhos), *parametros_ajustados)
        plt.plot(tamanhos, curva_ajustada, label=f'Curva Teórica')

    plt.xlabel('Tamanho da Entrada')
    plt.ylabel('Tempo (s)')
    plt.title(f'{nome_algoritmo}: Tempo Real vs Tempo Teórico')
    plt.legend()
    plt.savefig(f'{nome_algoritmo}_graph.png')
    plt.close()

def comparar_todos_algoritmos(resultados):
    plt.figure(figsize=(12, 8))
    
    for nome, tempos in resultados.items():
        plt.plot(tamanhos_entradas, tempos, label=nome, marker='o')
    
    plt.xlabel('Tamanho da Entrada')
    plt.ylabel('Tempo (s)')
    plt.title('Comparação de Todos os Algoritmos de Ordenação')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparacao_algoritmos.png')
    plt.close()

def salvar_tabela_tempos(resultados):
    with open('tabela_tempos.txt', 'w') as f:
        
        f.write('Tamanho')
        for nome in resultados.keys():
            f.write(f'\t{nome}')
        f.write('\n')
        
        
        for i, tamanho in enumerate(tamanhos_entradas):
            f.write(f'{tamanho}')
            for nome in resultados.keys():
                f.write(f'\t{resultados[nome][i]:.4f}')
            f.write('\n')

def gerar_arquivo_entrada(nome_arquivo, tamanho):
    with open(nome_arquivo, 'w') as arquivo:
        for _ in range(tamanho):
            arquivo.write(str(random.randint(1, tamanho)) + '\n')


tamanhos_entradas = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 16000, 20000, 25000]


for tamanho in tamanhos_entradas:
    gerar_arquivo_entrada(f'entrada_{tamanho}.txt', tamanho)


algoritmos = [
    (bubble_sort, 'Bubble Sort', funcao_ajuste_quadratica),
    (insertion_sort, 'Insertion Sort', funcao_ajuste_quadratica),
    (selection_sort, 'Selection Sort', funcao_ajuste_quadratica),
    (merge_sort, 'Merge Sort', funcao_ajuste_logaritmica),
    (quick_sort, 'Quick Sort', funcao_ajuste_logaritmica)
]

resultados = {}

for algoritmo, nome, funcao_ajuste in algoritmos:
    tempos_execucao_reais = []
    for tamanho in tamanhos_entradas:
        with open(f'entrada_{tamanho}.txt', 'r') as arquivo:
            valores = [int(linha.strip()) for linha in arquivo]

        print(f'Iniciando {nome} para entrada de tamanho {tamanho}')
        tempo_real, valores_ordenados = medir_tempo(algoritmo, valores)
        tempos_execucao_reais.append(tempo_real)
        print(f'{nome} para entrada de tamanho {tamanho} concluído. Tempo Real: {tempo_real:.4f} segundos\n')

        salvar_saida_ordenada(valores_ordenados, f'saida_ordenada_{nome.lower().replace(" ", "_")}_{tamanho}.txt')
    
    resultados[nome] = tempos_execucao_reais
    comparar_tempos(tempos_execucao_reais, None, tamanhos_entradas, nome, funcao_ajuste)

comparar_todos_algoritmos(resultados)
salvar_tabela_tempos(resultados)