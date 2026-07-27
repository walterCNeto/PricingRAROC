"""
Interest Rate Loan Pricing — RAROC (Consignado INSS)
Porte do app Shiny para Streamlit.  Autor do modelo: Walter Correa Neto (BFEng).

Roda de graça no Streamlit Community Cloud.  Arquivo único, sem banco de dados.
O padrão reproduz exatamente os números da calculadora Shiny; os "ajustes de
consistência" (opcionais) aplicam as correções dimensionais discutidas na validação.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------------------
#  NÚCLEO DO MODELO  (porte fiel de funcoes.R)
# ----------------------------------------------------------------------------

def calcular_rho(PD: float, prazo: int) -> float:
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    t1 = 0.03 * (1 - np.exp(-35 * PD_LT)) / (1 - np.exp(-35))
    t2 = 0.16 * (1 - (1 - np.exp(-35 * PD_LT)) / (1 - np.exp(-35)))
    return t1 + t2


def calculate_IRB_capital(EAD, LGD, PD, rho, prazo) -> float:
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    PD_LT = min(max(PD_LT, np.finfo(float).eps), 1 - np.finfo(float).eps)
    inner = (norm.ppf(PD_LT) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho)
    return EAD * LGD * (norm.cdf(inner) - PD_LT)  # já é UL (líquida de EL, via -PD)


def calculate_pmt(principal, rate, prazo) -> float:
    return principal * rate / (1 - (1 + rate) ** (-prazo))


def spread_el(PD, LGD, prazo, rrf) -> float:
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    ELR = PD_LT * LGD
    return (ELR / (1 - ELR)) * (1 + rrf)          # eq. 3.25 (corrigido: (1+rrf))


def spread_ul(EAD, LGD, PD, prazo, funding_cost, rho, premio_capital,
              di_futuro, di_atual, modo="fiel") -> float:
    K = calculate_IRB_capital(EAD, LGD, PD, rho, prazo)
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    ELR = PD_LT * LGD
    if modo == "premio":
        # recomendado: capital já rende a taxa livre de risco no numerador do RAROC,
        # então o spread cobre apenas o prêmio de capital exigido.
        excesso = premio_capital
    else:
        # fiel à Shiny atual: re = funding + prêmio, subtraindo di_futuro
        re = funding_cost + premio_capital
        excesso = re - di_futuro
    return (K / EAD) * excesso / (1 - ELR)


def calculate_RAROC(interest_rate, EAD, LGD, PD, rho, funding_cost, prazo,
                    pis_cofins, comissao, custos_adm, ir_cs, fator_pond, di_atual,
                    comissao_integral=False):
    im = (1 + interest_rate) ** (1 / 12) - 1
    fm = (1 + funding_cost) ** (1 / 12) - 1
    capital = calculate_IRB_capital(EAD, LGD, PD, rho, prazo)
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    EL = PD_LT * LGD * EAD
    R = calculate_pmt(EAD, im, prazo) * prazo - EAD
    C = calculate_pmt(EAD, fm, prazo) * prazo - EAD
    PB = R - C
    pisc = PB * pis_cofins
    fator_custo = 1.0 if comissao_integral else (12 * (1 / prazo))
    LAIR = PB - pisc - EL - (comissao * EAD * fator_custo) - (custos_adm * EAD * fator_custo)
    RGO = LAIR - LAIR * ir_cs
    piso = 8 / 100 * fator_pond * EAD
    cap = max(capital, piso)
    num = RGO + cap * ((1 + di_atual) ** (prazo / 12) - 1)
    return (num / cap) * (12 / prazo)


def otimizar_taxa(EAD, LGD, PD, rho, funding_cost, prazo, alvo_anual,
                  pis_cofins, comissao, custos_adm, ir_cs, fator_pond, di_atual,
                  base_consistente=False, comissao_integral=False):
    # alvo: base fiel (acumulado na vida) vs consistente (anual = anual)
    target = alvo_anual if base_consistente else (alvo_anual * prazo / 12)

    def f(i):
        return calculate_RAROC(i, EAD, LGD, PD, rho, funding_cost, prazo,
                               pis_cofins, comissao, custos_adm, ir_cs, fator_pond,
                               di_atual, comissao_integral) - target
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return float("nan")


# ----------------------------------------------------------------------------
#  INTERFACE
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Interest Rate Loan Pricing", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f6f8fc; }
  .block-container { padding-top: 1.6rem; max-width: 1150px; }
  h1, h2, h3 { color: #1b2456; }
  div[data-testid="stMetric"] {
      background:#01003c; border-radius:12px; padding:16px 20px; color:#fff;
      box-shadow:0 4px 14px rgba(20,20,60,.18);
  }
  div[data-testid="stMetric"] label { color:#cadcfc !important; }
  div[data-testid="stMetricValue"] { color:#fff !important; font-size:2.2rem; }
  .foot { color:#6b7280; font-size:.85rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Interest Rate Loan Pricing")
st.caption("Financial Engineering for Bankers · precificação RAROC ajustada ao risco")

# ---- entradas ----
st.subheader("Parâmetros da operação")
c1, c2, c3, c4 = st.columns(4)
with c1:
    ID = st.text_input("ID Client", placeholder="Ex: xxx.xxx.xxx-xx")
    EAD = st.number_input("Loan Amount (R$)", 500.0, 100000.0, 5000.0, 500.0)
    PD = st.number_input("PD", 0.005, 0.99, 0.017, 0.001, format="%.3f")
    pis = st.number_input("PIS/COFINS Tax", 0.0, 0.50, 0.0465, 0.001, format="%.4f")
with c2:
    prazo = st.select_slider("Loan Term (months)", options=list(range(12, 61, 6)), value=24)
    LGD = st.number_input("LGD", 0.10, 0.99, 0.75, 0.05)
    di_futuro = st.number_input("Funding Rate – Duration (% a.a.)", 0.1, 50.0, 12.99, 0.01)
    comissao = st.number_input("Fee (comissão)", 0.0, 0.50, 0.06, 0.01)
with c3:
    di_atual = st.number_input("Funding Rate – Risk Free (% a.a.)", 0.1, 50.0, 11.70, 0.01)
    funding_pct = st.number_input("Funding Transfer Price Factor (%)", 100.0, 500.0, 128.0, 10.0)
    custo_capital = st.number_input("Capital Premium (p.p.)", 0.0, 15.0, 5.0, 0.5)
    custos_adm = st.number_input("Admin Costs", 0.0, 0.50, 0.01, 0.01)
with c4:
    fator_pond = st.number_input("Risk Weighting (FPR)", 0.0, 0.99, 0.50, 0.01)
    ir_cs = st.number_input("IR/CS Tax", 0.0, 0.80, 0.40, 0.01)

with st.expander("⚙️ Ajustes de consistência (opcional — o padrão reproduz a Shiny atual)"):
    st.caption("Correções dimensionais discutidas na validação. Desligadas = mesmo número do app atual.")
    ac1, ac2 = st.columns(2)
    modo_ul = "premio" if ac1.checkbox("Spread UL só sobre o prêmio de capital (recomendado)") else "fiel"
    base_cons = ac2.checkbox("Base temporal consistente (RAROC anual = alvo anual)")
    comissao_int = ac1.checkbox("Comissão/custos integrais (one-time, sem fator 12/n)")

# ---- cálculo ----
rho = calcular_rho(PD, prazo)
funding_cost = (di_futuro / 100) * (funding_pct / 100)
premio = custo_capital / 100
s_el = spread_el(PD, LGD, prazo, funding_cost)
s_ul = spread_ul(EAD, LGD, PD, prazo, funding_cost, rho, premio,
                 di_futuro / 100, di_atual / 100, modo=modo_ul)
alvo = funding_cost + s_el + s_ul
K = calculate_IRB_capital(EAD, LGD, PD, rho, prazo)

taxa_aa = otimizar_taxa(EAD, LGD, PD, rho, funding_cost, prazo, alvo,
                        pis, comissao, custos_adm, ir_cs, fator_pond, di_atual / 100,
                        base_consistente=base_cons, comissao_integral=comissao_int)
taxa_am = (1 + taxa_aa) ** (1 / 12) - 1 if np.isfinite(taxa_aa) else float("nan")

# ---- resultado ----
st.subheader("Resultado")
if np.isfinite(taxa_aa):
    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric("Interest Rate (% a.a.)", f"{taxa_aa*100:.2f}%")
    m2.metric("Interest Rate (% a.m.)", f"{taxa_am*100:.2f}%")
    with m3:
        st.markdown("**Composição (anual)**")
        st.dataframe(pd.DataFrame({
            "Componente": ["Funding cost", "Spread EL", "Spread UL", "RAROC alvo", "K_IRB (capital)"],
            "Valor": [f"{funding_cost*100:.2f}%", f"{s_el*100:.2f}%", f"{s_ul*100:.2f}%",
                      f"{alvo*100:.2f}%", f"R$ {K:,.2f}"],
        }), hide_index=True)
else:
    st.error("Não foi possível resolver a taxa com esses parâmetros. Revise as entradas.")

# ---- export excel ----
if np.isfinite(taxa_aa):
    export = pd.DataFrame([{
        "ID Client": ID, "Simulation Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Loan Amount (EAD)": EAD, "Term (months)": prazo, "PD": PD, "LGD": LGD,
        "Funding Rate - Duration (% a.a.)": di_futuro, "Funding Rate - Risk Free (% a.a.)": di_atual,
        "Funding Transfer Price (%)": funding_pct, "Capital Premium (p.p.)": custo_capital,
        "PIS/COFINS Tax (%)": pis, "Commission Fee (%)": comissao, "Admin Costs (%)": custos_adm,
        "IR + CS Tax (%)": ir_cs, "Risk Weight (FPR)": fator_pond,
        "Funding Cost (%)": funding_cost, "Spread EL (%)": s_el, "Spread UL (%)": s_ul,
        "K_IRB (R$)": K, "Min Interest Rate (a.a.)": taxa_aa, "Min Interest Rate (a.m.)": taxa_am,
        "UL mode": modo_ul, "Consistent time base": base_cons, "One-time costs": comissao_int,
        "Author": "WalterCN - Banking Financial Engineering - BFEng",
    }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        export.to_excel(w, index=False, sheet_name="Simulation")
    st.download_button("⬇️ Baixar Excel", buf.getvalue(),
                       file_name=f"Simulation_{datetime.now():%Y-%m-%d}_BFEng.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown(
    '<div class="foot">paper: '
    '<a href="https://rpubs.com/WalterCN/RarocPricing" target="_blank">Link</a> · '
    'contato: <a href="mailto:walter.correa.neto@gmail.com">walter.correa.neto@gmail.com</a></div>',
    unsafe_allow_html=True)
