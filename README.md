#  Algoritmo UMDA Discreto — OneMax

Este proyecto implementa el **Univariate Marginal Distribution Algorithm (UMDA)** en su versión discreta utilizando Python.  
El objetivo es optimizar la función **OneMax**, que busca maximizar la cantidad de bits con valor 1 en un vector binario.

---

## Descripción general

El algoritmo UMDA es un modelo de estimación de distribuciones que aprende las probabilidades marginales de las mejores soluciones (élite) y genera nuevas poblaciones a partir de ellas.  
Este enfoque reemplaza los operadores clásicos de cruzamiento y mutación por un modelo probabilístico simple basado en la distribución **Bernoulli**.

---

## Características principales
- Implementación **totalmente en Python**.
- Evaluación experimental con múltiples configuraciones:
  - Tamaños de problema: `n = 20, 50, 100`
  - Tamaños de población: `N = 30, 50, 100`
  - Porcentaje de élite: `40%`
  - Ejecuciones por configuración: `30`
- Métricas de desempeño:
  - Mejor fitness promedio
  - Desviación estándar
  - Tasa de éxito
  - Generaciones promedio hasta convergencia

---

## Resultados destacados
| n | N | Éxito (%) | Mejor Prom | Gen. Conv. |
|---|---|------------|-------------|-------------|
| 20 | 50 | 100.0 | 20.00 / 20 | 4.5 |
| 50 | 100 | 100.0 | 50.00 / 50 | 8.6 |
| 100 | 100 | 96.7 | 99.97 / 100 | 13.7 |

> A mayor tamaño de población, el algoritmo logra mayor estabilidad y tasa de éxito en la convergencia.

---

## Requisitos
```bash
pip install scipy
