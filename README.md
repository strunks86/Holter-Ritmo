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
- Lista de exámenes como la de HScribe (0.7.0): se elige una vez la carpeta de los Holter (Examen → Elegir la carpeta de los Holter) y el programa la recorre, muestra ID/RUT, apellidos, nombre, estado, fecha y hora, fecha de nacimiento y grupo, con buscador, filtros y orden por columna; doble clic abre el examen. Aparece sola al abrir el programa. Lee también el repositorio de HScribe (carpetas Exam_N con slot_N).
- Paciente, diario, estados del examen (Adquirido, Editado, Revisado, Firmado), Buscar examen y Volver a escanear. El trabajo se guarda solo y se recupera al volver a abrir.
- Informe final estándar o condensado (portada, estadísticas, narrativo, perfiles, tendencias, plantillas, tiras e información completa) con vista previa, impresión y PDF; exporta latidos a CSV y estadísticas a XML.
- Qué funciones del manual de HScribe 6.4 tiene: [docs/INVENTARIO-HSCRIBE.md](docs/INVENTARIO-HSCRIBE.md).
- Opciones tomadas de medilog Darwin2 (0.6.0): pestañas Arritmias (tendencias, bloques y resumen tabular de día y noche), Bandas, Ruido, Mín./Máx., Cascada, Auricular (PR), ECHOView, VFC completa, QT y ST y Medidor; herramientas de intervalo en el ECG, flujo de trabajo, atajos configurables, periodo de sueño y criterios de Darwin. Detalle: [docs/INVENTARIO-DARWIN2.md](docs/INVENTARIO-DARWIN2.md).
- Planilla de tiras del informe: normal, FC máx./mín., V, pares, tripletas, TV (la más rápida y la más larga), S, pares S, TSV (la más rápida y la más larga) y BAV de 2.º y 3.er grado.

## Desarrollo

```
pip install -r requirements.txt
python HolterLocal.py            # requiere node en el PATH
python HolterLocal.py --autoprueba salida_prueba ruta/al/examen.hea
```

Herramienta de apoyo: toda conclusión clínica la revisa un médico.
