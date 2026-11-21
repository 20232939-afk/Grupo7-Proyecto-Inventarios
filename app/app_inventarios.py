"""
Aplicación Streamlit – Simulador de escenarios de inventario (EC2 – Parte A, Grupo 7).

Ejecutar desde la carpeta del proyecto:
    streamlit run app/app_inventarios.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import motor
from motor import sim

st.set_page_config(page_title="Simulador de inventarios – Grupo 7", page_icon="📦", layout="wide")
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c"]
INS = motor.cargar_insumos()

# ---------------------------------------------------------------------------
# Barra lateral: parámetros globales
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    autor = st.text_input("Nombre del estudiante (encabezado)", "Sebastián Tomaya Ojeda")
    variante_principal = st.radio("Política a destacar", ["Ajustada", "Clásica"], index=0,
                                  help="Ajustada por undershoot = política principal aprobada (Etapa 4b). "
                                       "Clásica = fórmula del informe original, como referencia.")
    with st.form("parametros"):
        st.subheader("Demanda (Etapa 3)")
        demanda = st.number_input("Demanda diaria pronosticada d̂", value=round(INS["demanda"], 2), min_value=1.0, step=0.5,
                                  help=f"Modelo {INS['modelo']}: equivale a la media histórica.")
        sigma = st.number_input("σ del error de pronóstico", value=round(INS["sigma"], 2), min_value=0.1, step=0.5)
        st.subheader("Política de reposición 🟠")
        dias_pedido = st.number_input("Q = días de demanda por pedido", value=7.0, min_value=1.0, step=1.0)
        dias_inicial = st.number_input("Inventario inicial = días de demanda + SS", value=7.0, min_value=0.0, step=1.0)
        st.subheader("Simulación")
        n_rep = st.select_slider("Réplicas Monte Carlo", options=[200, 500, 1000, 2000], value=1000)
        semilla = st.number_input("Semilla", value=42, step=1)
        with st.expander("💰 Costos (supuestos parametrizables)"):
            st.caption("El dataset no contiene costos. Valores por defecto de la Etapa 4b.")
            c_unit = st.number_input("Costo unitario (S/)", value=round(INS["costo_unitario"], 2), min_value=0.0)
            c_tasa = st.number_input("Tasa anual de mantener (%)", value=INS["tasa_mantener"] * 100, min_value=0.0) / 100
            c_ped = st.number_input("Costo por pedido (S/)", value=INS["costo_pedido"], min_value=0.0)
            c_falt = st.number_input("Costo por unidad no atendida (S/)", value=round(INS["costo_faltante"], 2), min_value=0.0)

        st.subheader("Escenarios")
        escenarios = []
        for i, (ns_def, nombre_def) in enumerate([(90.0, "Escenario A – NS 90%"), (95.0, "Escenario B – NS 95%"), (99.0, "Escenario C – NS 99%")]):
            with st.expander(nombre_def, expanded=False):
                nombre = st.text_input("Nombre", nombre_def, key=f"n{i}")
                ns = st.slider("Nivel de servicio por ciclo objetivo (%)", 80.0, 99.5, ns_def, 0.5, key=f"ns{i}")
                lt = st.slider("Lead time (días)", 1, 5, 2, key=f"lt{i}")
                fd = st.slider("Factor de demanda (1 = pronóstico)", 0.7, 1.3, 1.0, 0.05, key=f"fd{i}")
                escenarios.append(motor.Escenario(nombre, ns / 100, lt, fd))
        st.form_submit_button("▶️ Ejecutar simulación", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="background:linear-gradient(90deg,#0f3d5e,#1f6f8b);padding:18px 24px;border-radius:10px;margin-bottom:12px">
  <div style="color:#fff;font-size:28px;font-weight:700">📦 Simulador de escenarios de inventario</div>
  <div style="color:#e6f2f7;font-size:17px;margin-top:4px">Estudiante: <b>{autor}</b></div>
  <div style="color:#cfe3ec;font-size:14px">Modelamiento Predictivo de Datos · Grupo 7 · EC2 – Parte A · Universidad de Lima</div>
</div>""", unsafe_allow_html=True)
st.caption(f"Caso: supermercado (supermarket_sales.csv). Horizonte simulado: {INS['fechas'][0]:%d/%m/%Y} – "
           f"{INS['fechas'][-1]:%d/%m/%Y} ({len(INS['fechas'])} días). Inventario **agregado** (todas las líneas). "
           f"Política destacada: **{variante_principal}**.")


# ---------------------------------------------------------------------------
# Simulación (con caché)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def correr(esc_tuplas, demanda, sigma, n_dias, n_rep, semilla, dias_pedido, dias_inicial):
    esc = [motor.Escenario(*t) for t in esc_tuplas]
    pols, k = motor.simular_escenarios(esc, demanda, sigma, n_dias, n_rep, semilla, dias_pedido, dias_inicial)
    return pols, k


esc_tuplas = tuple((e.nombre, e.nivel_servicio, e.lead_time, e.factor_demanda) for e in escenarios)
if len({e.nombre for e in escenarios}) < 3:
    st.error("Los tres escenarios deben tener nombres distintos."); st.stop()
with st.spinner(f"Simulando {len(escenarios)} escenarios × 2 políticas × {n_rep} réplicas…"):
    pols, kpis = correr(esc_tuplas, demanda, sigma, len(INS["fechas"]), n_rep, int(semilla), dias_pedido, dias_inicial)
resumen = motor.resumen_con_ic(kpis)
cst = motor.costos(kpis, c_unit, c_tasa, c_ped, c_falt, len(INS["fechas"]))
nombres = [e.nombre for e in escenarios]
cortos = [n.split("–")[0].strip() for n in nombres]          # etiquetas cortas para ejes
color_de = dict(zip(nombres, COLORES))
ORDEN_KPI = list(motor.KPI_ETIQUETAS.values())


def valor(esc, var, kpi):
    return resumen.query("Escenario == @esc and Variante == @var and KPI == @kpi")["Media"].iloc[0]


tabs = st.tabs(["📊 Comparación de escenarios", "📈 Inventario proyectado", "📦 Distribuciones",
                "⚖️ Ajustada vs. clásica", "🧮 Parámetros y fórmulas", "⬇️ Descargas"])

# ---------------------------------------------------------------------------
# 1. Comparación
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader(f"Indicadores clave – política {variante_principal} (media de {n_rep} réplicas)")
    cols = st.columns(3)
    for col, e in zip(cols, escenarios):
        p = pols[(e.nombre, variante_principal)]
        with col:
            st.markdown(f"<div style='border-left:6px solid {color_de[e.nombre]};padding-left:10px'>"
                        f"<b>{e.nombre}</b><br><small>NS {e.nivel_servicio:.1%} · LT {e.lead_time} d · "
                        f"demanda ×{e.factor_demanda:.2f}</small></div>", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric("Stock de seguridad", f"{p.stock_seguridad:,.1f}")
            b.metric("ROP", f"{p.rop:,.1f}")
            a.metric("Fill rate", f"{valor(e.nombre, variante_principal, 'Fill rate (%)'):.2f} %")
            b.metric("Servicio por ciclo", f"{valor(e.nombre, variante_principal, 'Nivel de servicio por ciclo (%)'):.1f} %",
                     delta=f"{valor(e.nombre, variante_principal, 'Nivel de servicio por ciclo (%)') - e.nivel_servicio * 100:+.1f} pp",
                     delta_color="normal", help="Diferencia en puntos porcentuales respecto del objetivo del escenario.")
            a.metric("Días con quiebre", f"{valor(e.nombre, variante_principal, 'Días con quiebre'):.2f}")
            b.metric("Unid. no atendidas", f"{valor(e.nombre, variante_principal, 'Unidades no atendidas'):,.1f}")
            a.metric("Inventario promedio", f"{valor(e.nombre, variante_principal, 'Inventario promedio disponible'):,.1f}")
            b.metric("Rotación anual", f"{valor(e.nombre, variante_principal, 'Rotación anualizada'):.1f}")
            a.metric("Cobertura (días)", f"{valor(e.nombre, variante_principal, 'Días de inventario (cobertura)'):.2f}")
            b.metric("Pedidos", f"{valor(e.nombre, variante_principal, 'Pedidos emitidos'):.1f}")
            costo = cst.query("Escenario == @e.nombre and Variante == @variante_principal")["Costo_total"].mean()
            st.metric("Costo trimestral estimado 🟠", f"S/ {costo:,.0f}")

    st.divider()
    kpis_graf = ["Fill rate (%)", "Nivel de servicio por ciclo (%)", "Días con quiebre", "Unidades no atendidas",
                 "Inventario promedio disponible", "Rotación anualizada"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=kpis_graf, vertical_spacing=.16)
    sub = resumen[resumen["Variante"] == variante_principal]
    for i, kpi in enumerate(kpis_graf):
        s = sub[sub["KPI"] == kpi].set_index("Escenario").loc[nombres]
        fig.add_bar(x=cortos, y=s["Media"], marker_color=COLORES, showlegend=False,
                    error_y=dict(type="data", symmetric=False, array=s["P95"] - s["Media"], arrayminus=s["Media"] - s["P5"]),
                    text=[f"{v:,.1f}" for v in s["Media"]], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="white"), row=i // 3 + 1, col=i % 3 + 1)
        if kpi == "Nivel de servicio por ciclo (%)":
            fig.add_scatter(x=cortos, y=[e.nivel_servicio * 100 for e in escenarios], mode="markers", name="Objetivo",
                            marker=dict(symbol="line-ew-open", size=40, color="red", line=dict(width=3)), row=1, col=2)
    fig.update_layout(height=620, margin=dict(t=70, b=10), title_text="Media y rango P5–P95 entre réplicas",
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    fig.update_xaxes(tickangle=0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Tabla comparativa** (IC 95 % = precisión de la media; P5–P95 = variabilidad de un trimestre individual)")
    tabla = sub.copy()
    tabla["Media [IC 95 %]"] = tabla.apply(lambda r: f"{r['Media']:,.2f} [{r['IC95 inf']:,.2f} – {r['IC95 sup']:,.2f}]", axis=1)
    tabla["P5 – P95"] = tabla.apply(lambda r: f"{r['P5']:,.1f} – {r['P95']:,.1f}", axis=1)
    st.dataframe(tabla.pivot(index="KPI", columns="Escenario", values="Media [IC 95 %]").loc[ORDEN_KPI, nombres], use_container_width=True)
    st.dataframe(tabla.pivot(index="KPI", columns="Escenario", values="P5 – P95").loc[ORDEN_KPI, nombres], use_container_width=True)

    st.markdown("**Costos trimestrales estimados 🟠** (supuestos editables en la barra lateral)")
    ct = cst[cst["Variante"] == variante_principal].groupby("Escenario")[["Costo_mantener", "Costo_ordenar", "Costo_faltante", "Costo_total"]].mean().loc[nombres]
    st.dataframe(ct.style.format("S/ {:,.0f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Trayectoria
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Inventario proyectado día a día (una réplica)")
    c1, c2 = st.columns([1, 3])
    rep = c1.number_input("Réplica a visualizar", 0, n_rep - 1, 0, help="Réplica 0 = primera trayectoria de la semilla.")
    var_t = c1.radio("Política", ["Ajustada", "Clásica"], index=0 if variante_principal == "Ajustada" else 1, key="vt")
    c2.info("Área de color = inventario **disponible** al cierre · gris = inventario **en tránsito** · "
            "línea discontinua = ROP · punteada = stock de seguridad · ✖ = día con quiebre.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=nombres)
    detalles = {}
    for i, e in enumerate(escenarios):
        p = pols[(e.nombre, var_t)]
        d = motor.trayectoria(p, demanda * e.factor_demanda, sigma, INS["fechas"], int(rep), int(semilla), n_rep)
        detalles[e.nombre] = d
        fig.add_scatter(x=d["Fecha"], y=d["Inventario_final_disponible"], fill="tozeroy", mode="lines", name="Disponible",
                        line=dict(color=COLORES[i]), showlegend=i == 0, legendgroup="d", row=i + 1, col=1)
        fig.add_scatter(x=d["Fecha"], y=d["Inventario_final_disponible"] + d["En_transito_cierre"], fill="tonexty", mode="lines",
                        name="En tránsito", line=dict(color="lightgray", width=0), fillcolor="rgba(180,180,180,.45)",
                        showlegend=i == 0, legendgroup="t", row=i + 1, col=1)
        fig.add_hline(y=p.rop, line_dash="dash", line_color="black", row=i + 1, col=1,
                      annotation_text=f"ROP {p.rop:.0f}", annotation_position="top left")
        fig.add_hline(y=p.stock_seguridad, line_dash="dot", line_color="red", row=i + 1, col=1,
                      annotation_text=f"SS {p.stock_seguridad:.0f}", annotation_position="bottom left")
        q = d[d["Quiebre"] == 1]
        fig.add_scatter(x=q["Fecha"], y=np.zeros(len(q)), mode="markers", marker=dict(symbol="x", size=11, color="red"),
                        name="Quiebre", showlegend=i == 0, legendgroup="q", row=i + 1, col=1,
                        hovertext=[f"No atendidas: {v:.0f}" for v in q["No_atendidas"]])
    fig.update_layout(height=850, margin=dict(t=40), legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)
    resumen_rep = pd.DataFrame({n: {"Días con quiebre": int(d["Quiebre"].sum()), "Unidades no atendidas": d["No_atendidas"].sum(),
                                    "Fill rate %": d["Ventas"].sum() / d["Demanda"].sum() * 100,
                                    "Pedidos": int((d["Pedido_emitido"] > 0).sum()),
                                    "Inventario promedio": d["Inventario_final_disponible"].mean()} for n, d in detalles.items()})
    st.dataframe(resumen_rep.style.format("{:,.2f}"), use_container_width=True)
    with st.expander("Ver detalle diario"):
        esc_sel = st.selectbox("Escenario", nombres)
        st.dataframe(detalles[esc_sel].round(2), use_container_width=True, height=380)

# ---------------------------------------------------------------------------
# 3. Distribuciones
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"Variabilidad entre {n_rep} trimestres simulados – política {variante_principal}")
    kd = kpis[kpis["Variante"] == variante_principal]
    c = st.columns(3)
    for col, (k, t) in zip(c, [("Fill_rate_%", "Fill rate (%)"), ("Dias_con_quiebre", "Días con quiebre"),
                                ("Unidades_no_atendidas", "Unidades no atendidas")]):
        f = px.box(kd, x="Escenario", y=k, color="Escenario", color_discrete_map=color_de, points=False,
                   category_orders={"Escenario": nombres}, title=t)
        f.update_layout(showlegend=False, height=380, xaxis_title=None, yaxis_title=None)
        col.plotly_chart(f, use_container_width=True)
    prob = kd.groupby("Escenario")["Dias_con_quiebre"].apply(lambda s: (s > 0).mean() * 100).loc[nombres]
    st.markdown("**Probabilidad de tener al menos un día con quiebre en el trimestre:** " +
                " · ".join(f"{n}: **{v:.1f} %**" for n, v in prob.items()))

# ---------------------------------------------------------------------------
# 4. Ajustada vs clásica
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Política ajustada por undershoot (principal) vs. clásica (referencia)")
    st.markdown("La fórmula **clásica** supone revisión continua. Con revisión **diaria**, cuando se detecta que la posición "
                "bajó del ROP ya está en promedio **E[U]** unidades por debajo (*undershoot*), y el nivel de servicio real "
                "cae bajo el objetivo. La **ajustada** incorpora ese undershoot en el ROP.")
    k_comp = ["Nivel de servicio por ciclo (%)", "Fill rate (%)", "Días con quiebre", "Unidades no atendidas", "Inventario promedio disponible"]
    fig = make_subplots(rows=1, cols=5, subplot_titles=["Servicio por ciclo (%)", "Fill rate (%)", "Días con quiebre",
                                                         "Unid. no atendidas", "Inventario promedio"])
    for j, kpi in enumerate(k_comp):
        for var, colr in [("Clásica", "#9e9e9e"), ("Ajustada", "#7b3fa0")]:
            s = resumen.query("Variante == @var and KPI == @kpi").set_index("Escenario").loc[nombres]
            fig.add_bar(x=cortos, y=s["Media"], name=var, marker_color=colr,
                        showlegend=j == 0, legendgroup=var, row=1, col=j + 1)
        if j == 0:
            fig.add_scatter(x=cortos, y=[e.nivel_servicio * 100 for e in escenarios],
                            mode="markers", name="Objetivo", marker=dict(symbol="line-ew-open", size=30, color="red", line=dict(width=3)), row=1, col=1)
    fig.update_layout(barmode="group", height=440, margin=dict(t=90),
                      legend=dict(orientation="h", y=1.25, x=0.5, xanchor="center"))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
    comp = (resumen[resumen["KPI"].isin(k_comp)].pivot_table(index=["Escenario", "KPI"], columns="Variante", values="Media")
            .reindex(pd.MultiIndex.from_product([nombres, k_comp], names=["Escenario", "KPI"])))
    comp["Diferencia (Aj − Cl)"] = comp["Ajustada"] - comp["Clásica"]
    st.dataframe(comp.style.format("{:,.2f}"), use_container_width=True)
    ctot = cst.groupby(["Escenario", "Variante"])["Costo_total"].mean().unstack().loc[nombres]
    ctot["Ahorro de la ajustada"] = ctot["Clásica"] - ctot["Ajustada"]
    st.markdown("**Costo total trimestral 🟠 (supuestos)**")
    st.dataframe(ctot.style.format("S/ {:,.0f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Parámetros y fórmulas
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Parámetros de cada escenario")
    filas = []
    for e in escenarios:
        for var in ["Ajustada", "Clásica"]:
            p = pols[(e.nombre, var)]
            eu, vu = sim.undershoot_esperado(p.demanda_diaria, p.sigma_error)
            filas.append({"Escenario": e.nombre, "Política": var, "NS objetivo": f"{p.nivel_servicio:.1%}", "z": p.z,
                          "Lead time": p.lead_time, "d̂": p.demanda_diaria, "σe": p.sigma_error,
                          "E[U]": eu if var == "Ajustada" else 0.0, "σ protección": p.sigma_lt,
                          "Stock de seguridad": p.stock_seguridad, "ROP": p.rop, "Q": p.q, "Inventario inicial": p.inventario_inicial})
    st.dataframe(pd.DataFrame(filas).style.format(precision=2), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Política clásica (referencia)**")
        st.latex(r"z=\Phi^{-1}(NS)\qquad SS = z\,\sigma_e\sqrt{LT}\qquad ROP=\hat d\,LT+SS")
        st.markdown("**Política ajustada por undershoot (principal)**")
        st.latex(r"E[U]=\frac{\hat d^{2}+\sigma_e^{2}}{2\hat d}\qquad Var[U]=\frac{\hat d^{2}+3\sigma_e^{2}}{3}-E[U]^2")
        st.latex(r"ROP=\hat d\,LT+E[U]+z\sqrt{\sigma_e^{2}\,LT+Var[U]}\qquad SS = ROP-\hat d\,LT")
    with c2:
        st.markdown("**Indicadores**")
        st.latex(r"\text{Fill rate}=\frac{\sum \text{ventas}}{\sum \text{demanda}}")
        st.latex(r"\text{NS por ciclo}=1-\frac{\text{ciclos con quiebre}}{\text{ciclos evaluados}}")
        st.latex(r"\text{Rotación anual}=\frac{\sum\text{ventas}}{\overline{\text{inventario}}}\cdot\frac{365}{91}")
        st.latex(r"\text{Cobertura (días)}=\frac{\overline{\text{inventario}}}{\text{ventas diarias promedio}}")
        st.markdown("**Secuencia diaria:** recepción al inicio → venta = mín(disponible, demanda) → al cierre, si "
                    "disponible + en tránsito ≤ ROP se pide Q, que llega al inicio del día t + LT + 1.")

    st.markdown("**Origen de cada dato**")
    st.dataframe(pd.DataFrame([
        ("Demanda diaria d̂ = 61.91", "🔵 Calculado", "Pronóstico ARIMA(0,0,0)+c validado (Etapa 3); equivale a la media histórica"),
        ("σe = 22.38", "🔵 Calculado", "RMSE de validación con origen móvil h=1 (Etapa 3)"),
        ("Nivel de servicio, lead time, Q, inventario inicial", "🟠 Supuesto", "El dataset no registra inventarios ni proveedores"),
        ("Demanda simulada", "🟣 Simulado", "Normal(d̂, σe), entera, ≥ 0; semilla fija; mismas trayectorias en todos los escenarios"),
        ("Costos", "🟠 Supuesto", "El dataset no contiene costos (cogs = precio × cantidad)"),
    ], columns=["Elemento", "Tipo", "Fuente"]), use_container_width=True, hide_index=True)
    st.warning("**Limitaciones:** (1) inventario **agregado** de las seis líneas: el SS agregado es ≈ 44 % de la suma de SS por línea, "
               "por lo que los KPI son optimistas frente a una gestión por SKU; (2) lead time determinístico; (3) demanda normal "
               "independiente día a día; (4) ventas usadas como proxy de demanda; (5) costos supuestos: los montos son escenarios, no ahorros reales.")

# ---------------------------------------------------------------------------
# 6. Descargas
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Descargar resultados de esta simulación")
    param_df = pd.DataFrame([{**pols[(e.nombre, v)].to_dict(), "Escenario_app": e.nombre, "Variante": v}
                             for e in escenarios for v in ["Ajustada", "Clásica"]])
    tablas = {"kpi_replicas": kpis, "kpi_resumen_ic": resumen, "costos_replicas": cst, "parametros": param_df,
              "trayectoria_replica": pd.concat([d.assign(Escenario=n, Variante=var_t, Replica=int(rep)) for n, d in detalles.items()])}
    c = st.columns(len(tablas) + 1)
    for col, (n, t) in zip(c, tablas.items()):
        col.download_button(f"{n}.csv", t.to_csv(index=False).encode("utf-8-sig"), f"{n}.csv", "text/csv", use_container_width=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for n, t in tablas.items():
            t.to_excel(w, sheet_name=n[:31], index=False)
    c[-1].download_button("📗 Excel completo", buf.getvalue(), "simulacion_inventarios.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.divider()
st.caption(f"Simulación: {n_rep} réplicas · semilla {int(semilla)} · motor `scripts/simulador_inventario.py` "
           "(mismo código de los notebooks 04 y 04b; verificado con `scripts/test_app_motor.py`).")
