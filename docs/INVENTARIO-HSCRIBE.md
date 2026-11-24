# Funciones del manual de HScribe 6.4 y su estado en HolterLocal 0.5.0

Este inventario compara el manual del usuario de HScribe 6.4 con HolterLocal. El diseño de HolterLocal es propio; no usa logos ni marcas del fabricante.

Estados:
- **Ya estaba**: la función existía en la versión 0.4.2.
- **Nuevo 0.5.0**: se agregó en esta versión.
- **No aplica**: la función no tiene sentido en una app local de ritmo, y se explica por qué.

## Exámenes, paciente y flujo de trabajo (capítulos 7, 9, 11 y 13)

| Función del manual | Estado | Dónde está en HolterLocal |
|---|---|---|
| Abrir un examen importado | Ya estaba | Examen → Abrir examen (WFDB, EDF/BDF, ISHNE, texto) y Abrir carpeta de examen HScribe |
| Información del paciente (nombre, ID, fecha de nacimiento, sexo, médico, indicaciones, medicamentos, notas) | Nuevo 0.5.0 | Examen → Información del paciente |
| Buscar examen, con filtro por estado y por fecha (todos, hoy, última semana) | Nuevo 0.5.0 | Examen → Buscar examen (lista local de exámenes abiertos en este computador) |
| Estados Adquirido, Editado, Revisado y Firmado | Nuevo 0.5.0 | Pasa a Editado al corregir; Examen → Finalizar examen pide el estado, quién revisó y quién firma |
| Finalizar el examen, con vista previa e impresión según el estado | Nuevo 0.5.0 | Examen → Finalizar examen |
| Guardar el trabajo y retomarlo | Nuevo 0.5.0 | Se guarda solo en la carpeta de datos del usuario (nunca junto al examen original). Al volver a abrir, se reaplican las correcciones |
| Volver a escanear: excluir canales y acortar la duración | Nuevo 0.5.0 | Examen → Volver a escanear |
| Criterios de escaneo (prematuridad S, pausa, ST, taquicardia, bradicardia) | Ya estaba | Pestaña Criterios |
| Criterios nuevos: depresión y elevación del ST por separado, pausa solo N‑N, FC solo con latidos normales, FC media sin pausas, marcapasos y su frecuencia | Nuevo 0.5.0 | Pestaña Criterios |
| Diario del paciente | Nuevo 0.5.0 | Editar → Lista del diario. Aparece en el perfil, en la barra de sucesos, en las tiras y en el informe |
| Ajustes del informe: institución, contacto, logotipo, médico que firma y resumen narrativo editable | Nuevo 0.5.0 | Editar → Ajustes del informe |
| Base de datos en red, usuarios, contraseñas y permisos por rol | No aplica | La app es local y de un solo usuario. Los permisos los da la cuenta de Windows |
| Lista de trabajo (MWL), HL7, DICOM, intercambio XML o PDF por carpetas | No aplica (en parte) | Sí hay exportación a PDF, CSV y XML desde el menú Examen. No hay integración con el sistema del hospital |
| Preparar y borrar la grabadora H3+ o la tarjeta H12+ | No aplica | Se hace con el software del fabricante. HolterLocal lee los exámenes ya adquiridos |
| Exámenes de 12 derivaciones y su vista | No aplica | En el hospital se usa un Holter de 3 canales. La app muestra todos los canales que tenga el examen |

## Revisión del análisis (capítulo 10)

| Función del manual | Estado | Dónde está en HolterLocal |
|---|---|---|
| Perfil horario con columnas de ritmo, ectopia, pausas, ST, VFC y QT; clic en una celda para ir al suceso | Nuevo 0.5.0 | Pestaña Perfil, con filtros Todo, General, Ritmo y Estimulado. Se pueden cambiar o restaurar valores y ocultar columnas. Usa periodos de 4 h si el examen dura más de 48 h |
| Prospectiva: ECG que avanza solo y se detiene en los sucesos elegidos, con varias velocidades e InstaPágina | Nuevo 0.5.0 | Pestaña Prospectiva |
| Superposición de latidos | Nuevo 0.5.0 | Pestaña Superposición |
| Plantillas (morfologías): reetiquetar, combinar, borrar, ver los latidos de cada una | Ya estaba en parte; ampliado en 0.5.0 | Pestaña Plantillas. Ahora permite elegir varias, combinarlas y borrarlas |
| Tendencias de FC, ectopia, ST, QT y VFC | Nuevo 0.5.0 | Pestaña Tendencias |
| Histogramas de RR, N‑N, FC, QT y PR, con clic para ir al latido | Nuevo 0.5.0 | Pestaña Histogramas |
| Tiras de ECG (manuales y automáticas), su orden, edición y anotación | Nuevo 0.5.0 | Pestaña Tiras (botón Agregar automáticas) y la herramienta Tira en el ECG |
| Resumen de estadísticas editable, con grupos que se pueden ocultar | Nuevo 0.5.0 | Pestaña Resumen |
| Diagnóstico, notas y conclusiones | Nuevo 0.5.0 | Pestaña Resumen |
| Lista de sucesos con hora, duración y paso de uno a otro | Ya estaba | Pestaña Eventos y botones de navegación del ECG |
| Sucesos definidos por el usuario (3 tipos con nombre propio) | Nuevo 0.5.0 | Herramienta Suceso en el ECG y Editar → Editar etiquetas de sucesos |
| Ajustar el inicio y fin de un suceso, borrarlo, y editar el nivel de ST | Nuevo 0.5.0 | Menú contextual sobre el suceso en el ECG |
| FA: agregar y quitar tramos a mano | Nuevo 0.5.0 | Herramienta Suceso y clic derecho en el ECG |
| Artefacto: marcar y devolver al análisis, incluidos los artefactos automáticos | Nuevo 0.5.0 | Herramienta Suceso, botón Artefacto en Tiras, y Editar → Deshacer artefacto |
| Etiquetas de latido: Normal, Supraventricular, Ventricular, Fusión, Aberrante, Desconocido, Bloqueo de rama, R sobre T, Interpolado, Escape ventricular, Estimulado, Auricular estimulado, Doble estimulado y Artefacto | Nuevo 0.5.0 (antes había 10) | Menú Etiqueta, teclas N S V F A U B T I E P R C D, y clic derecho |
| Insertar y borrar latidos | Nuevo 0.5.0 | Mayúsculas + letra para insertar, Supr para borrar |
| Selección múltiple de latidos | Nuevo 0.5.0 | Mayúsculas + clic y Ctrl + clic |
| Calibradores de tiempo y amplitud, con marcha de calibrador | Nuevo 0.5.0 | Herramienta Calibrador |
| Ganancia, velocidad del papel (mm/s) y duración de la página | Ya estaba en parte; ampliado en 0.5.0 | Barra del ECG: Página y Ganancia, y el menú Formato |
| Elegir canales, filtro y vista de contexto | Ya estaba en parte; ampliado en 0.5.0 | Casillas de canales, Filtro, Vista → Contexto y Vista → Solo ECG o ventana dividida |
| Ir a una hora determinada | Nuevo 0.5.0 | Navegar → Ir a una hora |
| Ocultar pestañas | Nuevo 0.5.0 | Menú Pestañas (se guarda por examen) |
| Atajos de teclado (F7, F8, Alt+número, Tab, flechas) | Nuevo 0.5.0 | Ayuda → Atajos de teclado |
| Imprimir la pantalla | Nuevo 0.5.0 | Examen → Imprimir pantalla |
| Histograma de espigas de marcapasos | No aplica por ahora | Los exámenes exportados no traen las espigas del marcapasos. Se puede agregar si el exportador las entrega |

## Informe final (capítulo 12)

| Función del manual | Estado | Dónde está en HolterLocal |
|---|---|---|
| Informe estándar y condensado, y tipos de informe guardados | Nuevo 0.5.0 | Examen → Imprimir informe |
| Secciones: portada con datos del paciente, estadísticas, resumen narrativo, perfiles, tendencias, plantillas, tiras e información completa | Nuevo 0.5.0 | Se eligen con casillas en el diálogo del informe |
| Información completa (todas las páginas de ECG de las horas y canales elegidos) | Nuevo 0.5.0 | Diálogo del informe |
| Vista previa, imprimir y guardar como PDF | Ya estaba solo el PDF del resumen; completo en 0.5.0 | Diálogo del informe |
| Firma y estado del examen en la portada | Nuevo 0.5.0 | Portada del informe |

## Lo que HolterLocal tiene y HScribe no

- Análisis siempre multicanal, con todos los canales a la vez.
- Motor holterai (modo mixto), con sus resultados en el resumen.
- Validación contra las anotaciones de las bases públicas (pestaña Validación).
