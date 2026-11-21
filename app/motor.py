"""
Motor de la aplicación de escenarios de inventario (EC2 – Parte A).
Lógica sin interfaz: la usa app_inventarios.py y se prueba en scripts/test_app_motor.py.
Toda la simulación se delega en scripts/simulador_inventario.py (mismo código de los notebooks 04 y 04b).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import simulador_inventario as sim  # noqa: E402

VARIANTES = {"Ajustada": True, "Clásica": False}      # Ajustada = principal (Etapa 4b); Clásica = referencia


@dataclass
class Escenario:
    nombre: str
    nivel_servicio: float
    lead_time: int = 2
    factor_demanda: float = 1.0      # 1.0 = pronóstico validado; permite estresar la demanda


def cargar_insumos() -> dict:
    """Lee el pronóstico validado (Etapa 3) y los supuestos de costo (Etapa 4b)."""
    par = pd.read_csv(ROOT / "outputs" / "etapa3" / "parametros_para_inventario.csv").set_index("Parámetro")["Valor"]
    pron = pd.read_csv(ROOT / "outputs" / "etapa3" / "pronostico_validado_diario.csv", parse_dates=["Fecha"])
    cos = pd.read_csv(ROOT / "outputs" / "etapa4b" / "supuestos_costo_base.csv").iloc[0]
    return {
        "modelo": str(par["Modelo seleccionado"]),
        "demanda": float(par["Demanda diaria pronosticada"]),
        "sigma": float(par["σ error de pronóstico (RMSE rolling h=1)"]),
        "fechas": pd.DatetimeIndex(pron["Fecha"]),
        "costo_unitario": float(cos["costo_unitario"]),
        "tasa_mantener": float(cos["tasa_mantener_anual"]),
        "costo_pedido": float(cos["costo_por_pedido"]),
        "costo_faltante": float(cos["costo_por_unidad_no_atendida"]),
    }


def politicas(escenarios: list[Escenario], demanda: float, sigma: float, dias_pedido: float,
              dias_inicial: float) -> dict[tuple[str, str], sim.Politica]:
    pols = {}
    for e in escenarios:
        for var, ajuste in VARIANTES.items():
            pols[(e.nombre, var)] = sim.calcular_politica(
                e.nivel_servicio, demanda * e.factor_demanda, sigma, lead_time=e.lead_time,
                dias_pedido=dias_pedido, dias_inicial=dias_inicial,
                escenario=f"{e.nombre} | {var}", ajuste_undershoot=ajuste)
    return pols


def simular_escenarios(escenarios: list[Escenario], demanda: float, sigma: float, n_dias: int,
                       n_replicas: int, semilla: int, dias_pedido: float = 7, dias_inicial: float = 7):
    """
    Devuelve (politicas, kpis por réplica). Cada escenario usa trayectorias de demanda generadas con la
    MISMA semilla (números aleatorios comunes): si dos escenarios comparten factor de demanda, ven
    exactamente la misma demanda, y las diferencias se deben solo a la política.
    """
    pols = politicas(escenarios, demanda, sigma, dias_pedido, dias_inicial)
    filas = []
    for e in escenarios:
        M = sim.generar_demanda(demanda * e.factor_demanda, sigma, n_dias, n_replicas, seed=semilla)
        for var in VARIANTES:
            p = pols[(e.nombre, var)]
            k = sim.montecarlo([p], M)
            k["Escenario"], k["Variante"] = e.nombre, var
            filas.append(k)
    return pols, pd.concat(filas, ignore_index=True)


def trayectoria(p: sim.Politica, demanda: float, sigma: float, fechas, replica: int, semilla: int,
                n_replicas: int) -> pd.DataFrame:
    """Detalle diario de una réplica concreta (misma matriz de demanda que la simulación)."""
    M = sim.generar_demanda(demanda, sigma, len(fechas), n_replicas, seed=semilla)
    det, _ = sim.simular(p, M[replica], fechas)
    return det


KPI_ETIQUETAS = {
    "Fill_rate_%": "Fill rate (%)",
    "Nivel_servicio_ciclo_%": "Nivel de servicio por ciclo (%)",
    "Dias_con_quiebre": "Días con quiebre",
    "Unidades_no_atendidas": "Unidades no atendidas",
    "Inventario_promedio_disponible": "Inventario promedio disponible",
    "Inventario_promedio_en_transito": "Inventario promedio en tránsito",
    "Pedidos_emitidos": "Pedidos emitidos",
    "Rotacion_anualizada": "Rotación anualizada",
    "Dias_de_inventario": "Días de inventario (cobertura)",
    "Undershoot_promedio": "Undershoot promedio",
}


def resumen_con_ic(kpis: pd.DataFrame) -> pd.DataFrame:
    """Media, IC 95 % de la media y P5–P95. El nivel de servicio por ciclo usa el estimador agrupado."""
    filas = []
    n = kpis.groupby(["Escenario", "Variante"]).size().iloc[0]
    t = stats.t.ppf(.975, n - 1)
    for (esc, var), g in kpis.groupby(["Escenario", "Variante"], sort=False):
        for k, etq in KPI_ETIQUETAS.items():
            x = g[k].dropna()
            media = x.mean()
            if k == "Nivel_servicio_ciclo_%":
                media = (1 - g["Ciclos_con_quiebre"].sum() / g["Ciclos_evaluados"].sum()) * 100
            ee = x.std(ddof=1) / np.sqrt(len(x))
            filas.append({"Escenario": esc, "Variante": var, "KPI": etq, "Media": media,
                          "IC95 inf": x.mean() - t * ee, "IC95 sup": x.mean() + t * ee,
                          "P5": x.quantile(.05), "P95": x.quantile(.95)})
    return pd.DataFrame(filas)


def costos(kpis: pd.DataFrame, costo_unitario: float, tasa: float, costo_pedido: float,
           costo_faltante: float, dias: int) -> pd.DataFrame:
    c = sim.SupuestosCosto(costo_unitario, tasa, costo_pedido, costo_faltante, dias)
    out = sim.evaluar_costos(kpis, c)
    out["Variante"] = kpis["Variante"].values
    return out
