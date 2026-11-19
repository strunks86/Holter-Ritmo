# HolterLocal

Programa de escritorio para Windows que analiza Holter de ritmo en el propio computador, sin internet ni navegador.

## Descargar

En la pestaña **Actions** de este repositorio, abre la última ejecución verde de "Windows (.exe)" y descarga **HolterLocal-Windows**. Trae dos opciones:

- `HolterLocal-<versión>-instalador.exe`: instala el programa con acceso en el menú Inicio y en el escritorio. No pide permisos de administrador.
- `HolterLocal-<versión>-portable.zip`: descomprimir y abrir `HolterLocal.exe`.

## Qué hace

- Abre exámenes WFDB (PhysioNet), EDF/BDF, ISHNE, texto y horas crudas de Mortara H12+.
- Analiza todos los canales juntos: detección de latidos, calidad por segundo y canal, clasificación N/S/V/F/artefacto, FA, pausas, bradi y taquicardia, rachas, bigeminismo, ST, QT y variabilidad.
- Pestañas: Resumen, Eventos, Plantillas, Tendencias, Histogramas (RR y Lorenz), Criterios y Validación (si el examen trae anotaciones de referencia).
- Tira de ECG en papel milimetrado con calibrador, reetiquetado por teclado y exclusión de tramos con artefacto.
- Exporta el resumen a PDF y los latidos a CSV.

## Desarrollo

```
pip install -r requirements.txt
python HolterLocal.py            # requiere node en el PATH
python HolterLocal.py --autoprueba salida_prueba ruta/al/examen.hea
```

Herramienta de apoyo: toda conclusión clínica la revisa un médico.
