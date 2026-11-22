# Funciones del manual de medilog Darwin2 y su estado en HolterLocal 0.6.0

Este inventario compara el manual de usuario de medilog Darwin2 (Schiller, 2.511064 rev. h, en español) con HolterLocal. Lo que ya había (incluido lo agregado desde el manual de HScribe, ver [INVENTARIO-HSCRIBE.md](INVENTARIO-HSCRIBE.md)) se dejó tal cual. Aquí solo se agregó lo que faltaba. El diseño es propio; no usa logos ni marcas del fabricante.

Estados:
- **Ya estaba**: existía en la versión 0.5.0.
- **Nuevo 0.6.0**: se agregó en esta versión.
- **No agregado**: necesita cambiar el motor de análisis o datos que el Holter del hospital no registra, y se explica por qué.

Todo lo nuevo usa los canales que tenga el examen (los 3 en el Holter del hospital). Las tiras pequeñas muestran todos los canales por defecto.

## Planilla de tiras del informe (pedida por Jorge)

| Tira | Estado | Cómo se elige |
|---|---|---|
| Normal | Nuevo 0.6.0 | 7,5 s de día, solo con latidos normales, con los canales limpios y sin tramos excluidos. Se elige la FC más cercana a la mediana del registro |
| Frecuencia máxima y mínima | Ya estaba (también en la planilla) | Respeta el valor fijado a mano en Mín./Máx. |
| Extrasístole ventricular, par ventricular | Ya estaba (también en la planilla) | Cantidad elegible, por examen, cada 24 h o cada hora |
| Tripleta ventricular | Nuevo 0.6.0 | Racha ventricular de exactamente 3 latidos |
| Taquicardia ventricular, la más rápida y la más larga | Nuevo 0.6.0 | Solo episodios clasificados como TV (no RIVA ni ritmo idioventricular). Más larga = más latidos |
| Extrasístole supraventricular, par supraventricular | Ya estaba (también en la planilla) | |
| TSV más rápida y más larga | Nuevo 0.6.0 | Rachas supraventriculares. Se descartan las de más de 300 lpm, que casi siempre son latidos mal detectados |
| Bloqueo AV de 2.º y 3.er grado | Nuevo 0.6.0 | Mobitz I, Mobitz II, 2:1, avanzado y completo de holterai. No incluye el de 1.er grado |

Está en Tiras → Agregar automáticas → pestaña **Planilla del informe**. Todas vienen marcadas. Si otra lista pide el mismo episodio, sale una sola tira.

## Pestañas nuevas

| Función del manual (capítulo) | Estado | Dónde está en HolterLocal |
|---|---|---|
| Directorio de ruido: ruido potencial, exclusiones automáticas y manuales, motivos (límites excedidos, señal ruidosa, ECG de baja amplitud, patrón improbable), confirmar o devolver al análisis (4.2.11) | Nuevo 0.6.0 | Pestaña **Ruido**. Por defecto muestra solo lo que afecta a todos los canales, para no excluir tiempo que otro canal ve limpio |
| Escáner de mínimos y máximos: NN, RR y FC de 3 latidos; fijar el valor correcto si el primero es artefacto (4.2.12) | Nuevo 0.6.0 | Pestaña **Mín./Máx.** El valor fijado reemplaza al automático en el resumen y mueve las tiras de FC máx./mín. |
| Tendencias de arritmia, una fila por tipo y ordenadas por gravedad (4.2.13) | Nuevo 0.6.0 | Pestaña **Arritmias**, modo Tendencias. Clic en un suceso para ir al ECG; Tab recorre la fila |
| Resumen de arritmias en bloques de color (4.2.16) | Nuevo 0.6.0 | Pestaña **Arritmias**, modo Bloques |
| Resumen tabular por intervalos, día, noche y registro completo; cantidad, duración o porcentaje (4.2.17) | Nuevo 0.6.0 | Pestaña **Arritmias**, modo Tabular. Clic en una celda recorre esos sucesos |
| TV sostenida (30 s o más) y no sostenida por separado | Nuevo 0.6.0 | En Arritmias, Bandas y el resumen tabular |
| Directorio de bandas: tiras de cada suceso en orden cronológico, por longitud o por gravedad; las 4 más graves al informe; comentarios (4.2.20) | Nuevo 0.6.0 | Pestaña **Bandas**. La casilla «Al informe» crea o quita la tira |
| Vista Cascada del editor de plantillas (4.2.10) | Nuevo 0.6.0 | Pestaña **Cascada**: latidos alineados en la R, reetiquetado con la letra. Sirve para confirmar que las S son supraventriculares verdaderas |
| Análisis auricular: PR latido a latido en el tiempo, histograma y % de PR sobre 200 ms (4.2.18) | Nuevo 0.6.0 | Pestaña **Auricular**. Usa el PR que mide holterai (sin holterai lo avisa) |
| ECHOView: todo el registro como imagen de latidos apilados, para ver cambios de morfología y ruido (4.2.19) | Nuevo 0.6.0 | Pestaña **ECHOView**. Permite elegir latidos y reetiquetarlos o excluir un tramo |
| VFC completa: tabla por ventanas, dominio de la frecuencia (ULF, VLF, LF, HF, LF/HF) con Welch o Burg, espectrograma, histogramas, diferencial, triangular, TINN, SDANN‑i, SDNN‑i, índice circadiano, resumen diario (capítulo 5) | Nuevo 0.6.0 (la VFC básica ya estaba) | Pestaña **VFC**, con prefiltro configurable y exportación a CSV |
| Tabla de QT con Bazett, Fridericia, Framingham, Hodges, Pfeufer y Sagie; QT largo como suceso; elegir el intervalo de QTc correcto (4.2.21) | Nuevo 0.6.0 | Pestaña **QT y ST** |
| Tabla de episodios del ST (4.2.22) | Nuevo 0.6.0 (los episodios ya estaban) | Pestaña **QT y ST** |
| Medidor de latidos: PR, QRS, QT, QTc y ST J+80 de cada canal, medidos a mano con marcas que se arrastran y guardados (4.2.2) | Nuevo 0.6.0 | Pestaña **Medidor**. Sigue al latido elegido en el ECG |

Todas se pueden ocultar desde el menú Pestañas.

## ECG, menús y ajustes

| Función del manual | Estado | Dónde está en HolterLocal |
|---|---|---|
| Herramientas de selección: marcador e intervalos [marcador–actual], [inicio–actual] y [actual–final] (3.4) | Nuevo 0.6.0 | Clic derecho en el ECG → Intervalo: artefacto, devolver al análisis, FA, quitar FA, borrar latidos, etiquetar todos, comentario en el diario |
| Fijar FC máxima o mínima desde el ECG | Nuevo 0.6.0 | Clic derecho → Intervalo → Fijar este latido como FC máxima/mínima |
| Flujo de trabajo guiado y editable (3.2) | Nuevo 0.6.0 | Ajustes → Panel de flujo de trabajo (Ctrl+W). Se edita el orden de los pasos |
| Atajos de teclado configurables (3.6) | Nuevo 0.6.0 | Ajustes → Atajos de teclado |
| Periodo de sueño del diario (luces, dormir, despertar) para el día y la noche | Nuevo 0.6.0 | Ajustes → Periodo de sueño. Por defecto 23:00 a 07:00 |
| Árbol de configuración de arritmias con los valores de Darwin (bradicardia ≤45, taquicardia >130, TSVP ≥110 lpm y 3 latidos, TV ≥111 lpm, pausa 1880 ms) (7.4) | Nuevo 0.6.0 | Ajustes → Cargar los criterios de arritmia de Darwin2. Se revisan en Criterios antes de aplicar |
| Exportar e importar criterios; guardar como predeterminados (7.4) | Nuevo 0.6.0 | Menú Ajustes (archivo .json) |
| Zoom, regla, calibrador, cuadrícula, ganancia, velocidad, filtro de pantalla | Ya estaba | Menú Formato y barra del ECG |
| Informe a medida con secciones elegibles | Ya estaba | Secciones nuevas (apagadas por defecto): resumen tabular de día y noche, VFC completa, QT y QTc, episodios del ST, mediciones manuales |

## No agregado

| Función del manual | Por qué |
|---|---|
| Excluir ruido de un solo canal | El motor excluye tiempo, no canales. Para no perder latidos que otro canal ve bien, el directorio de ruido propone por defecto solo lo que afecta a todos los canales |
| Pausa distinta después de una V (2380 ms) | Necesita un criterio nuevo en el motor |
| Clases J, U1–U4, Z y P con subtipos del editor de plantillas | El motor usa sus propias clases; agregar estas exige cambiar el motor |
| Detector de ECG de baja amplitud como criterio del motor | Se muestra como motivo en el directorio de ruido, pero el motor no lo usa para excluir |
| EDR (respiración derivada del ECG), posición corporal, SpO2, presión arterial | El Holter del hospital no registra esos datos |
| 12 derivaciones | El hospital usa Holter de 3 canales |
| Histogramas de espigas de marcapasos (PM‑PM, PM‑R, fallos de captura y sensado detallados) | No hay detección de espigas en la señal exportada |
| Tiempo relativo, calibración del monitor y colores configurables | Se dejan para una versión siguiente |
| Base de datos, usuarios, HIS, formato .DAS | La app es local y de un solo usuario |
