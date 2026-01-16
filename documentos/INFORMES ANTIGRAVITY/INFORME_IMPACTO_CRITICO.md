# 🩸 INFORME DE IMPACTO CRÍTICO: "¿QUÉ SE ROMPE?"
**FECHA:** 14-ENE-2026
**OBJETIVO:** AUTOPSIA PREVENTIVA DE LOS EFECTOS SECUNDARIOS DEL "PLATINUM+ COMBO".

---

Manel, has pedido sangre y verdad. Aquí está.
Hemos arreglado la "fuga de datos" y la "falta de visión", pero al hacerlo, hemos roto otras cosas.
Este es el precio de la excelencia técnica.

## 1. CEGUERA MACROESTRUCTURAL (LO MÁS GRAVE)
*   **QUÉ HEMOS HECHO:** Hemos quitado los datos `D1` para entrenar solo con `M5` "puro".
*   **QUÉ SE ROMPE:** **El contexto.** El modelo ahora es un francotirador que mira por una mira telescópica de 5 aumentos. Ve perfectamente el movimiento del precio en los últimos 300 minutos (60 velas x 5 min), pero **NO SABE** si el precio está chocando contra una resistencia histórica de hace 5 años.
*   **CONSECUENCIA:** Puede dar señal de `BUY` perfecta en técnica M5 justo debajo de un techo de hormigón D1. Se estrellará.

## 2. FRAGILIDAD PARAMÉTRICA (TRIPLE BARRIER)
*   **QUÉ HEMOS HECHO:** Hemos fijado `barrier_width = 1.5 * volatilidad`.
*   **QUÉ SE ROMPE:** **La adaptabilidad extrema.** 1.5 es un "número mágico".
    *   Si el mercado se vuelve **muy errático** (ruido puro), 1.5 será demasiado estrecho y entraremos en falsas rupturas.
    *   Si el mercado se vuelve **muy direccional** (tendencia suave), 1.5 será demasiado ancho y no entraremos nunca.
*   **CONSECUENCIA:** Necesitamos re-optimizar este 1.5 cada mes. Hoy funciona, el mes que viene puede ser basura.

## 3. DEPENDENCIA CRÍTICA DEL PROVEEDOR DE DATOS (VOLUMEN)
*   **QUÉ HEMOS HECHO:** Hemos metido `log_vol` (Volumen) como input clave.
*   **QUÉ SE ROMPE:** **La portabilidad.** El `tick_volume` no es volumen real de mercado, es "actividad del broker".
*   **CONSECUENCIA:** Si mañana cambias de broker, o tu broker cambia su feed de datos, el histograma de volumen cambiará. El modelo, entrenado con el perfil de volumen de "Broker A" (2020-2025), fallará estrepitosamente con "Broker B" o con un cambio en la API. Hemos atado el modelo a tu fuente de datos actual.

## 4. "SOBRE-CORRECCIÓN" DEL SCALER
*   **QUÉ HEMOS HECHO:** Ajuste estricto del Scaler solo en Train (80%).
*   **QUÉ SE ROMPE:** **La capacidad de reacción ante Cisnes Negros.**
*   **CONSECUENCIA:** Si en el futuro (Validation o Real) el precio o la volatilidad saltan a niveles nunca vistos en el 80% del Train (ej. Guerra Mundial III, oro a 4000), el Scaler comprimirá esos datos de forma extraña o los saturará. El modelo no sabrá qué hacer con valores fuera de escala (out-of-distribution).

## 5. EL RIESGO DE "SILENCIO DE RADIO" (HOLD ETERNO)
*   **QUÉ HEMOS HECHO:** Combinar Focal Loss + Triple Barrier es muy agresivo filtrando ruido.
*   **QUÉ SE ROMPE:** **La frecuencia operativa.**
*   **CONSECUENCIA:** Es posible que el modelo se vuelva **demasiado conservador**. Puede pasar de operar mal (48%) a no operar casi nunca (99% Hold), esperando la "señal perfecta" que solo ocurre una vez al mes. Esto mejora el accuracy teórico, pero mata la rentabilidad por inactividad.

---

## VEREDICTO
El sistema es ahora **mucho más robusto científicamente**, pero también **más frágil operativamente**.
Requiere vigilancia constante del parámetro `1.5` y de la calidad del volumen del broker.
**Ya no es un "juguete", es un Fórmula 1: corre más, pero si pisas una piedra, se desintegra.**
