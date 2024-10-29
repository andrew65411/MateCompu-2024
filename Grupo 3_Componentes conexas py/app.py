from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx

app = Flask(__name__)

# Función para graficar la matriz mostrando solo 0 y 1 usando seaborn
def graficar_matriz(matriz, titulo):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='binary', cbar=False, linewidths=.5, linecolor='black')
    plt.title(titulo)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# Función para graficar un grafo dirigido a partir de una matriz de adyacencia
def graficar_grafo_componentes(componentes, vertices_nombres):
    grafo_imagenes = []
    for i, componente in enumerate(componentes):
        G = nx.DiGraph()  # Grafo dirigido

        # Asegúrate de que solo agregamos los nodos de la componente
        for nodo in componente:
            G.add_node(vertices_nombres[nodo])  # Usar nombres de vértices en lugar de índices

        # Agregar aristas para crear un ciclo
        for j in range(len(componente)):
            G.add_edge(vertices_nombres[componente[j]], vertices_nombres[componente[(j + 1) % len(componente)]])

        # Posicionamiento
        pos = nx.circular_layout(G)  # Usar un diseño circular para los ciclos
        plt.figure(figsize=(6, 4))
        ax = plt.gca()  # Obtener el eje actual

        # Dibujar el grafo con el estilo de flechas dirigido
        nx.draw(G, pos, with_labels=True, arrows=True, node_size=1000, node_color="lightblue",
                font_size=10, edge_color='black', arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

        plt.title(f"Grafo de Componente Conexa {i + 1}")
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        grafo_imagenes.append(base64.b64encode(img.getvalue()).decode())

    return grafo_imagenes


def calcular_matriz_con_diagonal(matriz):
    np.fill_diagonal(matriz, 1)
    return matriz

def calcular_matriz_caminos(matriz):
    n = len(matriz)
    caminos = np.copy(matriz)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                caminos[i][j] = caminos[i][j] or (caminos[i][k] and caminos[k][j])
    return caminos

# Función para ordenar las filas según la cantidad de 1's
def ordenar_filas(matriz, nombres_vertices):
    conteo_filas = np.sum(matriz, axis=1)  # Cuenta la cantidad de conexiones en cada fila
    orden_filas = np.argsort(-conteo_filas)  # Ordena en base a la cantidad de conexiones, en orden descendente
    matriz_ordenada_filas = matriz[orden_filas]  # Reordena las filas de la matriz
    nombres_ordenados = [nombres_vertices[i] for i in orden_filas]  # Reordena los nombres de los vértices
    return matriz_ordenada_filas, orden_filas, nombres_ordenados

# Función para ordenar las columnas de acuerdo al mismo orden aplicado a las filas
def ordenar_columnas(matriz, orden_filas):
    matriz_ordenada_columnas = matriz[:, orden_filas]  # Reordena las columnas de la matriz
    return matriz_ordenada_columnas

def identificar_componentes_conexas_por_bloques(matriz):
    n = len(matriz)
    componentes = []
    i = 0
    
    while i < n:
        if matriz[i, i] == 1:
            componente = [i]
            for j in range(i + 1, n):
                if matriz[i, j] == 1 and matriz[j, i] == 1:
                    componente.append(j)
                else:
                    break
            componentes.append(componente)
            i += len(componente)  # Saltar a la siguiente fila después de la componente
        else:
            i += 1  # Continuar al siguiente índice si no hay un bloque

    # Imprimir las componentes de manera más clara
    for idx, comp in enumerate(componentes):
        print(f"Componente Conexa {idx + 1}: ", ", ".join(str(v) for v in comp))

    return componentes


@app.route('/', methods=['GET'])
def home():
    return render_template('inicio.html')

@app.route('/iniciar', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form['num_nodos'])
        metodo = request.form.get('metodo')

        # Obtener nombres de vértices
        nombres_vertices = request.form.get('nombres_vertices', '')
        if nombres_vertices:
            if ',' not in nombres_vertices:
                return render_template('index.html', error="Por favor, ingresa los nombres de los vértices separados por comas.", n=n)

            vertices_nombres = nombres_vertices.split(',')
            vertices_nombres = [nombre.strip() for nombre in vertices_nombres]
        else:
            vertices_nombres = [chr(65 + i) for i in range(n)]  # Nombres por defecto (A, B, C, ...)

        if n < 8 or n > 16:
            return render_template('index.html', error="El número de nodos debe estar entre 8 y 16.", n=n)

        if metodo == 'aleatorio':
            matriz = np.random.randint(2, size=(n, n))
            np.fill_diagonal(matriz, 0)
        else:
            matriz = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    matriz[i][j] = int(request.form[f'elemento_{i}_{j}'])

        # Paso 1: Matriz con Diagonal Asegurada
        matriz_diagonal = calcular_matriz_con_diagonal(matriz.copy())
        matriz_diagonal_img = graficar_matriz(matriz_diagonal, "Paso 1: Matriz con Diagonal Asegurada")

        # Paso 2: Matriz de Caminos
        matriz_caminos = calcular_matriz_caminos(matriz_diagonal.copy())
        matriz_caminos_img = graficar_matriz(matriz_caminos, "Paso 2: Matriz de Caminos")

        # Paso 3: Ordenar filas
        matriz_ordenada_filas, orden_filas, nombres_ordenados = ordenar_filas(matriz_caminos.copy(), vertices_nombres)
        matriz_ordenada_filas_img = graficar_matriz(matriz_ordenada_filas, "Paso 3: Filas Ordenadas según la Cantidad de 1's")

        # Paso 4: Ordenar columnas según el nuevo orden de las filas
        matriz_filas_columnas_ordenadas = ordenar_columnas(matriz_ordenada_filas, orden_filas)
        matriz_filas_columnas_ordenadas_img = graficar_matriz(matriz_filas_columnas_ordenadas, "Paso 4: Columnas Ordenadas según el Orden de las Filas")

        # Actualizar los nombres de vértices a los nombres ordenados
        vertices_nombres = nombres_ordenados

        # Identificar componentes conexas por bloques diagonales
        componentes = identificar_componentes_conexas_por_bloques(matriz_filas_columnas_ordenadas)

        # Grafo de componentes conexas
        grafo_graficas = graficar_grafo_componentes(componentes, vertices_nombres)

        return render_template(
            'resultado.html',
            matriz_diagonal_img=matriz_diagonal_img,
            matriz_caminos_img=matriz_caminos_img,
            matriz_ordenada_filas_img=matriz_ordenada_filas_img,
            matriz_filas_columnas_ordenadas_img=matriz_filas_columnas_ordenadas_img,
            componentes=componentes,
            grafo_graficas=grafo_graficas,
            vertices_nombres=vertices_nombres
        )

    return render_template('index.html', n=None)

@app.route('/integrantes', methods=['GET'])
def integrantes():
    return render_template('integrantes.html')

if __name__ == "__main__":
    app.run(debug=True)
