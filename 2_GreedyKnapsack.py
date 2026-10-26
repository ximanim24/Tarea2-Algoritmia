import matplotlib.pyplot as plt
import numpy as np

def greedy_knapsack(items, capacity, criterion="value_weight"):
    """
    items: lista de diccionarios con id, weight, value
    capacity: capacidad máxima de la mochila
    criterion: 'value_weight' (tradicional) o 'min_weight' (menor peso primero)
    """
    if criterion == "value_weight":
        # Ordenar por valor/peso descendente, y a igual ratio, mayor valor
        sorted_items = sorted(items, key=lambda x: (x['value'] / x['weight'], x['value']), reverse=True)
    elif criterion == "min_weight":
        # Ordenar por menor peso primero, y a igual peso, mayor valor
        sorted_items = sorted(items, key=lambda x: (x['weight'], -x['value']))
    else:
        sorted_items = items

    total_weight = 0
    total_value = 0
    selected_items = []

    for item in sorted_items:
        if total_weight + item['weight'] <= capacity:
            total_weight += item['weight']
            total_value += item['value']
            selected_items.append(item['id'])
            
    return selected_items, total_weight, total_value

if __name__ == '__main__':
    print("--- Ejecutando Ejercicio 2: Greedy Knapsack ---")
    
    # Instancia diseñada para que el criterio de menor peso falle en comparación con el óptimo/tradicional:
    # Capacidad de la mochila = 10
    # Objeto A: Peso = 2, Valor = 10  (Ratio = 5.0) -> Muy alto valor y liviano
    # Objeto B: Peso = 9, Valor = 18  (Ratio = 2.0) -> Pesado pero gran valor total
    # Criterio Menor Peso elegirá primero A (peso 2, valor 10), se llena la mochila (10-2=8 restante) y ya no cabe B. Valor total = 10.
    # Criterio Tradicional (Valor/Peso) elegirá A (ratio 5), y dejará B fuera por espacio. Pero si ajustamos los pesos:
    
    # Usemos una instancia clásica de contraejemplo:
    # Capacidad W = 50
    # Ítem 1: Peso = 10, Valor = 60  (Ratio = 6.0)
    # Ítem 2: Peso = 20, Valor = 100 (Ratio = 5.0)
    # Ítem 3: Peso = 30, Valor = 120 (Ratio = 4.0)
    # Si priorizamos menor peso: Ítem 1 (p=10, v=60) + Ítem 2 (p=20, v=100) -> Peso total = 30, Valor total = 160.
    # Pero si el caso pone un objeto muy pesado y valioso que el menor peso rechaza de entrada:
    
    capacity = 50
    items = [
        {'id': 'Item A', 'weight': 10, 'value': 60},
        {'id': 'Item B', 'weight': 45, 'value': 90},
        {'id': 'Item C', 'weight': 5,  'value': 10}
    ]
    # Menor peso ordenaría: C (p=5, v=10), A (p=10, v=60). Total peso = 15, Valor = 70.
    # Tradicional por Valor/Peso ordenaría: A (ratio 6), C (ratio 2), B (ratio 2). A(10,60) + C(5,10) = Peso 15, Valor 70.
    
    # Diseñemos un contraejemplo perfecto donde Menor Peso falle feo:
    # Capacidad = 10
    # Objeto 1: Peso = 2, Valor = 3  (Ratio = 1.5) -> Liviano, poco valor
    # Objeto 2: Peso = 9, Valor = 10 (Ratio = 1.11) -> Pesado, pero alto valor
    # Objeto 3: Peso = 9, Valor = 9  (Ratio = 1.0)
    
    capacity = 10
    items = [
        {'id': 'Obj_Liviano_BajoValor', 'weight': 2, 'value': 3},
        {'id': 'Obj_Pesado_AltoValor',  'weight': 9, 'value': 20},
    ]
    # Si usas Menor Peso: Elige Obj_Liviano (peso 2, valor 3). La capacidad restante es 8, por lo que Obj_Pesado no entra. Valor total = 3.
    # Si usas Valor/Peso: Obj_Liviano (ratio 1.5), Obj_Pesado (ratio 20/9 = 2.22). Elige Obj_Pesado primero (peso 9, valor 20). Caben 1 de espacio libre. Valor total = 20!
    
    sel_vw, w_vw, v_vw = greedy_knapsack(items, capacity, criterion="value_weight")
    sel_mp, w_mp, v_mp = greedy_knapsack(items, capacity, criterion="min_weight")
    
    print(f"Método Tradicional (Valor/Peso) -> Seleccionados: {sel_vw}, Capacidad: {w_vw}/{capacity}, Valor Total: {v_vw}")
    print(f"Método Modificado (Menor Peso)  -> Seleccionados: {sel_mp}, Capacidad: {w_mp}/{capacity}, Valor Total: {v_mp}")
    
    # --- Visualización ---
    methods = ['Valor / Peso (Tradicional)', 'Menor Peso (Modificado)']
    values = [v_vw, v_mp]
    weights = [w_vw, w_mp]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, values, width, label='Valor Total', color='#2ca02c')
    rects2 = ax.bar(x + width/2, weights, width, label='Capacidad Usada', color='#ff7f0e')
    
    ax.set_ylabel('Métrica', fontsize=11)
    ax.set_title('Comparativa de Métodos Voraces para la Mochila (Knapsack)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    plt.savefig('knapsack_comparison.png', dpi=300, bbox_inches='tight')
    print("\n¡Gráfica generada y guardada como 'knapsack_comparison.png'!")
    plt.show()