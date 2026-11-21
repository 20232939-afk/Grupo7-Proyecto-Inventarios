"""
Simulador de inventarios – Grupo 7 (Etapa 4).

Política de revisión diaria (s, Q) con punto de reorden sobre la POSICIÓN de inventario:
al cierre de cada día, si  disponible + en tránsito <= ROP  se emite un pedido de Q unidades
que llega al INICIO del día t + LT + 1, es decir, después de LT días completos de consumo.

Supuestos del modelo (documentados en el notebook 04):
- Ventas perdidas (lost sales): la demanda que no se atiende con el inventario disponible se pierde.
- Lead time determinístico.
- Demanda diaria ~ Normal(media, sigma_error), redondeada a unidades enteras y truncada en 0.

Políticas disponibles (ambas se conservan; decisión del grupo tras la validación 4b):
- AJUSTADA por undershoot  -> calcular_politica(..., ajuste_undershoot=True)   [POLÍTICA PRINCIPAL]
  Alcanza el nivel de servicio por ciclo objetivo con revisión diaria (validado en notebook 04b).
- CLÁSICA                  -> calcular_politica(..., ajuste_undershoot=False)  [REFERENCIA]
  Fórmula de revisión continua del informe original; con revisión diaria queda por debajo del objetivo.

LIMITACIONES (deben declararse en el informe y en la aplicación):
1. Inventario AGREGADO: una sola "unidad genérica" que suma las seis líneas de producto. No representa
   la reposición por SKU. Por agregación (risk pooling) el SS agregado es ≈ 44 % de la suma de SS por
   línea, por lo que los KPI son optimistas frente a una gestión por línea (notebook 04b, sección 7).
2. Lead time, Q, inventario inicial y nivel de servicio son SUPUESTOS: el dataset no registra inventarios.
3. Demanda normal i.i.d.; las ventas observadas se usan como proxy de la demanda (posible demanda censurada).
4. Costos (SupuestosCosto): el dataset NO contiene costos. Todos los valores son supuestos parametrizables;
   los resultados económicos son escenarios, no ahorros reales.

Este módulo lo usan los notebooks 04 / 04b y la aplicación Streamlit (Etapa 6).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

POLITICA_PRINCIPAL = "ajustada"     # decisión aprobada en la Etapa 4b
POLITICA_REFERENCIA = "clasica"


# ---------------------------------------------------------------------------
# Parámetros y política
# ---------------------------------------------------------------------------
@dataclass
class Politica:
    escenario: str
    nivel_servicio: float          # objetivo de nivel de servicio por ciclo (α)
    z: float
    demanda_diaria: float          # pronóstico (d̂)
    sigma_error: float             # σ del error de pronóstico diario
    lead_time: int                 # días
    sigma_lt: float                # σ del error durante el lead time
    stock_seguridad: float
    demanda_lead_time: float
    rop: float
    q: float                       # cantidad de pedido
    inventario_inicial: float
    perdida_esperada_ciclo: float  # E[unidades no atendidas por ciclo] = σ_LT · L(z)
    fill_rate_teorico: float       # β ≈ 1 − E[faltante]/Q

    def to_dict(self) -> dict:
        return asdict(self)


def perdida_normal_estandar(z: float) -> float:
    """Función de pérdida normal estándar L(z) = φ(z) − z·(1 − Φ(z))."""
    return float(stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))


def undershoot_esperado(media: float, sigma: float) -> tuple[float, float]:
    """
    Undershoot U en revisión periódica diaria: cuánto queda la posición por debajo del ROP cuando
    se dispara el pedido. Con demanda diaria D ~ N(μ, σ) (aprox. de renovación):
        E[U]  = E[D²] / (2·E[D]) = (σ² + μ²) / (2μ)
        E[U²] = E[D³] / (3·E[D]) = (μ² + 3σ²) / 3
    Devuelve (E[U], Var[U]).
    """
    eu = (sigma ** 2 + media ** 2) / (2 * media)
    eu2 = (media ** 2 + 3 * sigma ** 2) / 3
    return eu, eu2 - eu ** 2


def calcular_politica(nivel_servicio: float, demanda_diaria: float, sigma_error: float,
                      lead_time: int = 2, dias_pedido: float = 7, dias_inicial: float = 7,
                      escenario: str | None = None, ajuste_undershoot: bool = False) -> Politica:
    """
    Fórmula clásica (revisión continua):  SS = z·σ_e·√LT ;  ROP = d̂·LT + SS
    Con ajuste_undershoot=True (revisión diaria): la protección debe cubrir la demanda del lead time
    MÁS el undershoot U:  ROP = d̂·LT + E[U] + z·√(σ_e²·LT + Var[U]) ;  SS = ROP − d̂·LT
    """
    z = float(stats.norm.ppf(nivel_servicio))
    dlt = demanda_diaria * lead_time
    if ajuste_undershoot:
        eu, vu = undershoot_esperado(demanda_diaria, sigma_error)
        sigma_lt = float(np.sqrt(sigma_error ** 2 * lead_time + vu))
        ss = eu + z * sigma_lt
    else:
        sigma_lt = sigma_error * np.sqrt(lead_time)
        ss = z * sigma_lt
    q = demanda_diaria * dias_pedido
    perdida = sigma_lt * perdida_normal_estandar(z)
    return Politica(
        escenario=escenario or f"NS {nivel_servicio:.0%}",
        nivel_servicio=nivel_servicio, z=z, demanda_diaria=demanda_diaria, sigma_error=sigma_error,
        lead_time=int(lead_time), sigma_lt=sigma_lt, stock_seguridad=ss, demanda_lead_time=dlt,
        rop=dlt + ss, q=q, inventario_inicial=demanda_diaria * dias_inicial + ss,
        perdida_esperada_ciclo=perdida, fill_rate_teorico=1 - perdida / q)


# ---------------------------------------------------------------------------
# Demanda
# ---------------------------------------------------------------------------
def generar_demanda(media: float, sigma: float, n_dias: int, n_replicas: int, seed: int = 42) -> np.ndarray:
    """Matriz (n_replicas × n_dias) de demandas enteras ≥ 0. Misma semilla → mismas trayectorias."""
    rng = np.random.default_rng(seed)
    d = rng.normal(media, sigma, size=(n_replicas, n_dias))
    return np.clip(np.rint(d), 0, None)


# ---------------------------------------------------------------------------
# Simulación de una trayectoria
# ---------------------------------------------------------------------------
def simular(pol: Politica, demanda: np.ndarray, fechas: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, list]:
    """
    Secuencia de cada día t:
      1. Inicio del día: se reciben los pedidos programados para t.
      2. Se atiende la demanda: ventas = min(disponible, demanda); no atendidas = demanda − ventas.
      3. Cierre del día: posición = disponible + en tránsito. Si posición ≤ ROP se emite un pedido Q,
         que llega al inicio de t + LT + 1 (consume LT días completos de demanda mientras está en tránsito).
    Devuelve el detalle diario y la lista de pedidos (día_emisión, día_llegada).
    """
    demanda = np.asarray(demanda, dtype=float)
    n, lt = len(demanda), pol.lead_time
    llegadas = np.zeros(n + lt + 2)
    disponible, en_transito = float(pol.inventario_inicial), 0.0
    pedidos, filas = [], []

    for t in range(n):
        inv_inicio = disponible
        recibido = llegadas[t]
        disponible += recibido
        en_transito -= recibido
        disponible_venta = disponible

        d = demanda[t]
        venta = min(disponible, d)
        no_atendida = d - venta
        disponible -= venta

        posicion_antes = disponible + en_transito
        pedido = 0.0
        # tolerancia 1e-9: evita que un error de redondeo de punto flotante (p. ej. 175.88000000001)
        # impida disparar el pedido cuando la posición es matemáticamente igual al ROP
        if posicion_antes <= pol.rop + 1e-9:
            pedido = pol.q
            dia_llegada = t + lt + 1
            llegadas[dia_llegada] += pedido
            en_transito += pedido
            pedidos.append((t, dia_llegada))

        filas.append({
            "Dia": t + 1, "Inventario_inicio": inv_inicio, "Recepcion": recibido,
            "Disponible_para_venta": disponible_venta, "Demanda": d, "Ventas": venta,
            "No_atendidas": no_atendida, "Quiebre": int(no_atendida > 0),
            "Inventario_final_disponible": disponible, "Pedido_emitido": pedido,
            "Undershoot": max(pol.rop - posicion_antes, 0.0) if pedido else 0.0,
            "En_transito_cierre": en_transito, "Posicion_inventario": disponible + en_transito,
        })

    detalle = pd.DataFrame(filas)
    if fechas is not None:
        detalle.insert(0, "Fecha", pd.DatetimeIndex(fechas)[:n])
    return detalle, pedidos


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------
def calcular_kpis(detalle: pd.DataFrame, pedidos: list, lead_time: int) -> dict:
    n = len(detalle)
    demanda_total = detalle["Demanda"].sum()
    ventas_total = detalle["Ventas"].sum()
    inv_prom = detalle["Inventario_final_disponible"].mean()

    # Nivel de servicio por ciclo (α): ¿hubo faltante mientras el pedido estaba en tránsito?
    quiebre = detalle["Quiebre"].to_numpy()
    ciclos = [(e, l) for (e, l) in pedidos if l - 1 < n]          # ciclos cuyo lead time terminó en el horizonte
    ciclos_con_quiebre = sum(quiebre[e + 1:l].any() for (e, l) in ciclos)

    return {
        "Demanda_total": demanda_total,
        "Ventas_total": ventas_total,
        "Unidades_no_atendidas": detalle["No_atendidas"].sum(),
        "Fill_rate_%": ventas_total / demanda_total * 100 if demanda_total else np.nan,
        "Dias_con_quiebre": int(quiebre.sum()),
        "%_dias_sin_quiebre": (1 - quiebre.mean()) * 100,
        "Ciclos_evaluados": len(ciclos),
        "Ciclos_con_quiebre": int(ciclos_con_quiebre),
        "Nivel_servicio_ciclo_%": (1 - ciclos_con_quiebre / len(ciclos)) * 100 if ciclos else np.nan,
        "Inventario_promedio_disponible": inv_prom,
        "Inventario_promedio_en_transito": detalle["En_transito_cierre"].mean(),
        "Inventario_minimo": detalle["Inventario_final_disponible"].min(),
        "Inventario_final": detalle["Inventario_final_disponible"].iloc[-1],
        "Pedidos_emitidos": int((detalle["Pedido_emitido"] > 0).sum()),
        "Undershoot_promedio": detalle.loc[detalle["Pedido_emitido"] > 0, "Undershoot"].mean()
                               if (detalle["Pedido_emitido"] > 0).any() else np.nan,
        "Unidades_pedidas": detalle["Pedido_emitido"].sum(),
        "Rotacion_periodo": ventas_total / inv_prom if inv_prom else np.nan,
        "Rotacion_anualizada": ventas_total / inv_prom * 365 / n if inv_prom else np.nan,
        "Dias_de_inventario": inv_prom / (ventas_total / n) if ventas_total else np.nan,
    }


def montecarlo(politicas: list[Politica], matriz_demanda: np.ndarray) -> pd.DataFrame:
    """Simula cada política sobre las MISMAS trayectorias de demanda (números aleatorios comunes)."""
    filas = []
    for pol in politicas:
        for r, dem in enumerate(matriz_demanda):
            det, ped = simular(pol, dem)
            filas.append({"Escenario": pol.escenario, "Replica": r, **calcular_kpis(det, ped, pol.lead_time)})
    return pd.DataFrame(filas)


@dataclass
class SupuestosCosto:
    """
    Costos SUPUESTOS y parametrizables: el dataset no registra costos de compra, de mantener,
    de ordenar ni de faltante. Ningún valor por defecto debe presentarse como dato observado.
    """
    costo_unitario: float            # costo de compra por unidad (S/)
    tasa_mantener_anual: float       # costo de mantener inventario, % anual del costo unitario
    costo_por_pedido: float          # costo fijo por orden emitida (S/)
    costo_por_unidad_no_atendida: float  # margen perdido (+ penalidad opcional) por unidad no vendida
    dias_horizonte: int = 91


def evaluar_costos(kpis: pd.DataFrame, c: SupuestosCosto) -> pd.DataFrame:
    """
    Costo del trimestre por réplica:
      mantener = inventario promedio disponible × costo unitario × tasa anual × días/365
      ordenar  = pedidos emitidos × costo por pedido
      faltante = unidades no atendidas × costo por unidad no atendida
    El inventario en tránsito no se incluye: con Q fijo es prácticamente igual entre políticas.
    """
    out = kpis[["Escenario", "Replica"]].copy()
    out["Costo_mantener"] = (kpis["Inventario_promedio_disponible"] * c.costo_unitario
                             * c.tasa_mantener_anual * c.dias_horizonte / 365)
    out["Costo_ordenar"] = kpis["Pedidos_emitidos"] * c.costo_por_pedido
    out["Costo_faltante"] = kpis["Unidades_no_atendidas"] * c.costo_por_unidad_no_atendida
    out["Costo_total"] = out[["Costo_mantener", "Costo_ordenar", "Costo_faltante"]].sum(axis=1)
    return out


def resumir_montecarlo(kpis_replicas: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación y percentiles 5–95 de cada KPI por escenario (formato largo, apto para Power BI)."""
    largo = kpis_replicas.melt(id_vars=["Escenario", "Replica"], var_name="KPI", value_name="Valor")
    return (largo.groupby(["Escenario", "KPI"])["Valor"]
            .agg(Media="mean", Desv="std", P5=lambda s: s.quantile(.05), Mediana="median",
                 P95=lambda s: s.quantile(.95))
            .reset_index())
