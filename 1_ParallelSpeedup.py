import time
import random
import multiprocessing
import matplotlib.pyplot as plt

def monte_carlo_pi_part(num_samples):
    """Calcula puntos aleatorios dentro del círculo unitario para una subtarea."""
    inside = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            inside += 1
    return inside

def estimate_pi(num_processes, total_samples=20_000_000):
    samples_per_process = total_samples // num_processes
    
    start_time = time.perf_counter()
    
    if num_processes == 1:
        total_inside = monte_carlo_pi_part(total_samples)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(monte_carlo_pi_part, [samples_per_process] * num_processes)
        total_inside = sum(results)
        
    pi_approx = 4 * total_inside / total_samples
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    return elapsed, pi_approx

if __name__ == '__main__':
    # Requerido para multiprocessing en sistemas operativos como Windows
    multiprocessing.freeze_support()
    
    processes_list = [1, 2, 4, 8]
    total_samples = 20_000_000  # Gran volumen de datos para notar la aceleración
    speedups = []
    
    print("--- Ejecutando Ejercicio 1: Aceleración Paralela ---")
    
    # 1. Obtener tiempo base con 1 proceso
    t1, pi1 = estimate_pi(1, total_samples)
    speedups.append(1.0)
    print(f"1 Proceso:  {t1:.4f} seg | Speedup: 1.00x | Pi: {pi1}")
    
    # 2. Obtener tiempos para 2, 4 y 8 procesos
    for p in [2, 4, 8]:
        tp, pip = estimate_pi(p, total_samples)
        speedup = t1 / tp
        speedups.append(speedup)
        print(f"{p} Procesos: {tp:.4f} seg | Speedup: {speedup:.2f}x | Pi: {pip}")
        
    # --- 3. Visualización ---
    plt.figure(figsize=(8, 5))
    plt.plot(processes_list, speedups, marker='o', linestyle='-', color='#1f77b4', label='Speedup Real', linewidth=2.5, markersize=8)
    plt.plot(processes_list, processes_list, linestyle='--', color='#d62728', label='Speedup Ideal (Lineal)', linewidth=2)
    
    plt.title('Aceleración Paralela (Speedup) vs. Número de Procesos', fontsize=13, fontweight='bold')
    plt.xlabel('Número de Procesos ($p$)', fontsize=11)
    plt.ylabel('Aceleración ($S(p)$)', fontsize=11)
    plt.xticks(processes_list)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Guardar gráfica automáticamente
    plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')
    print("\n ¡Gráfica generada y guardada como 'speedup_plot.png'!")
    plt.show()