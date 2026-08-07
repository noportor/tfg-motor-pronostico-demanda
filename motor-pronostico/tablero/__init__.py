"""Tablero — visor de solo lectura sobre las corridas del experimento.

El pipeline es la única fuente de verdad: escribe artefactos en ``salidas*/`` y
este paquete solo los LEE y los presenta. Ningún número citable en la tesis
puede originarse aquí (RN-1/RN-6); si un dato no está en las salidas, el lugar
de calcularlo es el pipeline, no el tablero.
"""
