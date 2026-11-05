# app.py — 🧩 Rompecabezas cónico 3D: armar y desarmar
# ---------------------------------------------------------------
# App interactiva (Streamlit + Plotly) que modela un rompecabezas de
# madera tipo cono/piña: N tablillas ranuradas dispuestas radialmente
# que se encajan entre sí. Las piezas alternadas van invertidas para
# lograr el encaje "tejido" (la ranura de una recibe la pestaña de otra).
#
# Ejecutar:   pip install -r requirements.txt && streamlit run app.py
# Requisitos: streamlit>=1.30, plotly>=5.18, numpy>=1.24

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Rompecabezas cónico 3D", page_icon="🧩", layout="wide")

# ------------------------- parámetros de la pieza (unidades arbitrarias)
H       = 5.5    # altura total del rompecabezas ensamblado
W_OUT   = 1.0    # radio exterior en la base (define la silueta cónica)
W_TOP   = 0.12   # radio exterior en la punta
W_IN    = 0.06   # radio interior (borde cercano al eje)
T       = 0.34   # espesor máximo de la tablilla (se ajusta según N)
SLOT_F  = 0.58   # altura del centro de la ranura (fracción de H)
SLOT_HF = 0.075  # semialtura de la ranura (fracción de H)
SLOT_D  = 0.34   # profundidad de la ranura


def _radio_exterior(y: float) -> float:
    """Radio del borde inclinado a la altura y (silueta cónica)."""
    return W_OUT + (W_TOP - W_OUT) * (y / H)


def malla_pieza(espesor: float):
    """Malla de una tablilla: perfil 2D con ranura, extruido con espesor.

    El perfil (trapecio con ranura tallada desde el borde interior) se
    descompone en subpolígonos convexos para triangularlo de forma robusta.
    Devuelve (vertices (M,3), caras (F,3)); la altura queda en el eje Y local.
    """
    lo = SLOT_F * H - SLOT_HF * H
    hi = SLOT_F * H + SLOT_HF * H
    perfil = [
        (W_IN, 0.0),                   # 0 interior, base
        (W_OUT, 0.0),                   # 1 exterior, base
        (W_TOP, H),                      # 2 exterior, punta
        (W_IN, H),                       # 3 interior, punta
        (W_IN, hi),                      # 4 ranura, arranque superior
        (W_IN + SLOT_D, hi),             # 5 ranura, fondo superior
        (W_IN + SLOT_D, lo),             # 6 ranura, fondo inferior
        (W_IN, lo),                      # 7 ranura, arranque inferior
        (_radio_exterior(hi), hi),       # 8 borde inclinado, arriba de la ranura
        (_radio_exterior(lo), lo),       # 9 borde inclinado, abajo de la ranura
    ]
    subpoligonos = [
        [0, 1, 9, 7],        # bloque inferior a la ranura (convexo)
        [6, 9, 8, 5],         # banda exterior de la ranura (convexa)
        [4, 5, 8, 2, 3],      # bloque superior a la ranura (convexo)
    ]
    contorno = [0, 1, 9, 8, 2, 3, 4, 5, 6, 7]   # perímetro exterior completo

    n = len(perfil)
    verts = ([(x, y, -espesor / 2) for x, y in perfil]
             + [(x, y, +espesor / 2) for x, y in perfil])
    faces = []

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for s in subpoligonos:                                  # tapas
        for k in range(1, len(s) - 1):
            a, b, c = s[0], s[k], s[k + 1]
            if abs(area2(perfil[a], perfil[b], perfil[c])) < 1e-12:
                continue                                     # triángulo degenerado
            faces.append((a, b, c))       # tapa +Z
            faces.append((a, c, b))       # tapa −Z
    for i, j in zip(contorno, contorno[1:] + contorno[:1]):  # caras laterales
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


_CACHE = {}


def piezas_para(n: int):
    """Devuelve (v, f, v_invertida, f_invertida) para un juego de n piezas."""
    espesor = min(T, max(0.12, 2 * 0.45 * np.sin(np.pi / n) - 0.03))
    key = round(float(espesor), 4)
    if key not in _CACHE:
        v, f = malla_pieza(espesor)
        v_inv = v.copy()
        v_inv[:, 1] = H - v_inv[:, 1]          # pieza invertida (ranura abajo)
        _CACHE[key] = (v, f, v_inv, f[:, [0, 2, 1]])
    return _CACHE[key]


def colocar(vertices, angulo, separacion=0.0, elevacion=0.0):
    """Rota la pieza alrededor del eje central y la desplaza radialmente."""
    c, s = np.cos(angulo), np.sin(angulo)
    x = vertices[:, 0] * c + vertices[:, 2] * s + separacion * c
    y = -vertices[:, 0] * s + vertices[:, 2] * c - separacion * s
    z = vertices[:, 1] + elevacion
    return np.column_stack([x, y, z])


PALETAS = {
    "Roble":   ["#c9a06e", "#b98a55", "#d8b183", "#a9743f", "#c49a6c",
                "#8d5a2b", "#d2a878", "#9c6b3d"],
    "Nogal":   ["#7a4a24", "#8f5a30", "#6b3f1d", "#9c6b3d", "#5f381a",
                "#a5754a", "#734021", "#87522c"],
    "Arce":    ["#e0b884", "#d9ab6d", "#e8c79a", "#cf9c5c", "#c08c50",
                "#efd3a8", "#d5a97a", "#e2bf92"],
    "Colores": ["#e4572e", "#2e86ab", "#f6ae2d", "#16a085", "#7d3f7d",
                "#c0392b", "#27ae60", "#2980b9"],
}
VERDE_TREBOL = "#3f7d3f"


def figura(n, desarmado, colocadas, paleta, trebol):
    """Construye la escena 3D del rompecabezas en su estado actual."""
    v, f, v_inv, f_inv = piezas_para(n)
    colores = PALETAS[paleta]
    fig = go.Figure()
    for k in range(n):
        pv, pf = (v, f) if k % 2 == 0 else (v_inv, f_inv)
        sep = desarmado * (1.3 + 0.9 * (k % 2))    # las impares salen más lejos
        elev = desarmado * 0.8 * (k % 2)           # y también más arriba
        pv = colocar(pv, 2 * np.pi * k / n, sep, elev)
        puesta = k < colocadas
        color = VERDE_TREBOL if (trebol and k == 0) else colores[k % len(colores)]
        fig.add_trace(go.Mesh3d(
            x=pv[:, 0], y=pv[:, 1], z=pv[:, 2],
            i=pf[:, 0], j=pf[:, 1], k=pf[:, 2],
            color=color if puesta else "#9aa0a6",
            opacity=1.0 if puesta else 0.15,
            flatshading=True,
            name=f"Pieza {k + 1}",
            hovertemplate=f"<b>Pieza {k + 1}</b><extra></extra>",
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.2,
                          roughness=0.6, fresnel=0.05),
            lightposition=dict(x=150, y=150, z=300),
        ))
    fig.update_layout(
        height=660,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=2.6, y=1.6, z=0.9)),
        ),
        legend=dict(itemsizing="constant"),
    )
    return fig


# ------------------------------------------------------- interfaz de usuario
st.title("🧩 Rompecabezas cónico 3D — armar y desarmar")
st.caption(
    "Modelo paramétrico de un rompecabezas de madera tipo cono/piña: tablillas "
    "ranuradas dispuestas radialmente que se encajan entre sí. "
    "Arrastra para rotar · rueda para zoom · clic en la leyenda para ocultar piezas."
)

with st.sidebar:
    st.header("⚙️ Controles")
    n = st.slider("Número de piezas", 4, 12, 6, 2,
                  help="Usa un número par para que las ranuras se alternen perfecto.")
    desarmado = st.slider("Desarmado", 0.0, 1.0, 0.0, 0.01, format="%.0f%%",
                          help="0 % = ensamblado · 100 % = totalmente desarmado")
    colocadas = st.slider("Piezas colocadas", 0, n, n,
                          help="Construye el rompecabezas pieza por pieza.")
    animar = st.button("▶ Animar: armar y desarmar", use_container_width=True)
    paleta = st.selectbox("Acabado", list(PALETAS))
    trebol = st.checkbox("Pieza del trébol verde", value=True)

colocadas = min(colocadas, n)

c1, c2, c3 = st.columns(3)
c1.metric("Piezas", n)
c2.metric("Triángulos en la malla", n * len(piezas_para(n)[1]))
c3.metric("Estado", "Ensamblado" if (desarmado < 0.02 and colocadas == n) else "Parcial")

zona = st.empty()
zona.plotly_chart(figura(n, desarmado, colocadas, paleta, trebol),
                  use_container_width=True)

if animar:
    frames = list(np.linspace(0.0, 1.0, 26)) + list(np.linspace(1.0, 0.0, 26))[1:]
    for s in frames:
        zona.plotly_chart(figura(n, float(s), n, paleta, trebol),
                          use_container_width=True)
        time.sleep(0.04)

with st.expander("📦 Subir a GitHub + desplegar en Streamlit Cloud"):
    st.markdown(
        """
        **Estructura del repositorio**
        ```
        rompecabezas-3d/
        ├── app.py            ← este archivo
        ├── requirements.txt
        └── README.md
        ```

        **Pasos**
        1. Crea un repositorio en GitHub y sube los tres archivos.
        2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
        3. **New app** → elige repositorio, rama `main`, archivo `app.py` → **Deploy**.
        4. Tu app quedará pública en `https://<usuario>-<repo>.streamlit.app`.

        **Local:** `pip install -r requirements.txt` y luego `streamlit run app.py`.
        """
    )
