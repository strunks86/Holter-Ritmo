# HolterLocal

Programa de escritorio para Windows que analiza Holter de ritmo en el propio computador, sin internet ni navegador.

## Descargar

En la pestaña **Actions** de este repositorio, abre la última ejecución verde de "Windows (.exe)" y descarga **HolterLocal-Windows**. Trae dos opciones:

- `HolterLocal-<versión>-instalador.exe`: instala el programa con acceso en el menú Inicio y en el escritorio. No pide permisos de administrador.
- `HolterLocal-<versión>-portable.zip`: descomprimir y abrir `HolterLocal.exe`.

## Qué hace

- Abre exámenes WFDB (PhysioNet), EDF/BDF, ISHNE, texto y horas crudas de Mortara H12+.
- Analiza todos los canales juntos: detección de latidos, calidad por segundo y canal, clasificación N/S/V/F/artefacto, FA, pausas, bradi y taquicardia, rachas, bigeminismo, ST, QT y variabilidad.
- Pestañas: Perfil, Prospectiva, Tendencias, Superposición, Plantillas, Histogramas, Tiras, Resumen, Eventos, Criterios y Validación (si el examen trae anotaciones de referencia).
- ECG en papel milimetrado con calibradores, 14 etiquetas de latido, insertar y borrar latidos, sucesos del usuario, FA y artefactos a mano.
- Paciente, diario, estados del examen (Adquirido, Editado, Revisado, Firmado), Buscar examen y Volver a escanear. El trabajo se guarda solo y se recupera al volver a abrir.
- Informe final estándar o condensado (portada, estadísticas, narrativo, perfiles, tendencias, plantillas, tiras e información completa) con vista previa, impresión y PDF; exporta latidos a CSV y estadísticas a XML.
- Qué funciones del manual de HScribe 6.4 tiene: [docs/INVENTARIO-HSCRIBE.md](docs/INVENTARIO-HSCRIBE.md).

## Desarrollo

```
pip install -r requirements.txt
python HolterLocal.py            # requiere node en el PATH
python HolterLocal.py --autoprueba salida_prueba ruta/al/examen.hea
```

Herramienta de apoyo: toda conclusión clínica la revisa un médico.
