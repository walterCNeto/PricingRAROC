"""
Interest Rate Loan Pricing — RAROC (Consignado INSS)
Porte do app Shiny para Streamlit.  Modelo: Walter Correa Neto (BFEng).

Roda de graça no Streamlit Community Cloud.  Arquivo único, sem banco de dados.

Metodologia embarcada = "Consistente" (correções de validação):
  (1) spread UL apenas sobre o prêmio de capital;
  (2) base temporal anual = anual (sem o fator n/12 no alvo);
  (3) comissão/custos integrais (one-time), sem o fator 12/n.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------------------
#  NÚCLEO DO MODELO
# ----------------------------------------------------------------------------

def calcular_rho(PD, prazo):
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    t1 = 0.03 * (1 - np.exp(-35 * PD_LT)) / (1 - np.exp(-35))
    t2 = 0.16 * (1 - (1 - np.exp(-35 * PD_LT)) / (1 - np.exp(-35)))
    return t1 + t2


def calculate_IRB_capital(EAD, LGD, PD, rho, prazo):
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    PD_LT = min(max(PD_LT, np.finfo(float).eps), 1 - np.finfo(float).eps)
    inner = (norm.ppf(PD_LT) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho)
    return EAD * LGD * (norm.cdf(inner) - PD_LT)      # já é UL (líquida de EL, via -PD)


def calculate_pmt(principal, rate, prazo):
    return principal * rate / (1 - (1 + rate) ** (-prazo))


def spread_el(PD, LGD, prazo, rrf):
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    ELR = PD_LT * LGD
    return (ELR / (1 - ELR)) * (1 + rrf)              # eq. 3.25 corrigida: (1+rrf)


def spread_ul(EAD, LGD, PD, prazo, funding_cost, rho, premio_capital,
              di_futuro, di_atual, modo="premio"):
    K = calculate_IRB_capital(EAD, LGD, PD, rho, prazo)
    PD_LT = 1 - (1 - PD) ** (prazo / 12)
    ELR = PD_LT * LGD
    if modo == "premio":
        # recomendado: o capital já rende a taxa livre de risco no numerador do
        # RAROC, então o spread cobre apenas o prêmio de capital exigido.
        excesso = premio_capital
    else:
        # legado (Shiny): re = funding + prêmio, subtraindo di_futuro.
        excesso = (funding_cost + premio_capital) - di_futuro
    return (K / EAD) * excesso / (1 - ELR)


def calculate_RAROC(interest_rate, EAD, LGD, PD, rho, funding_cost, prazo,
                    pis_cofins, comissao, custos_adm, ir_cs, fator_pond, di_atual,
                    comissao_integral=True, tarifa=0.0):
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
    # tarifa = receita de originação (one-time); comissão/custos = despesa (one-time)
    LAIR = (PB - pisc - EL
            - (comissao * EAD * fator_custo) - (custos_adm * EAD * fator_custo)
            + (tarifa * EAD * fator_custo))
    RGO = LAIR - LAIR * ir_cs
    piso = 8 / 100 * fator_pond * EAD
    cap = max(capital, piso)
    num = RGO + cap * ((1 + di_atual) ** (prazo / 12) - 1)
    return (num / cap) * (12 / prazo)                 # RAROC anualizado [% a.a.]


def otimizar_taxa(EAD, LGD, PD, rho, funding_cost, prazo, alvo_anual,
                  pis_cofins, comissao, custos_adm, ir_cs, fator_pond, di_atual,
                  base_consistente=True, comissao_integral=True, tarifa=0.0):
    # alvo: consistente (anual = anual) vs legado (acumulado na vida, x n/12)
    target = alvo_anual if base_consistente else (alvo_anual * prazo / 12)

    def f(i):
        return calculate_RAROC(i, EAD, LGD, PD, rho, funding_cost, prazo,
                               pis_cofins, comissao, custos_adm, ir_cs, fator_pond,
                               di_atual, comissao_integral, tarifa) - target
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return float("nan")


def taxa_efetiva_ifrs9(EAD, PMT, prazo, tarifa, comissao, custos_adm):
    """TIR mensal que amortiza custos/receitas de originação (custo amortizado IFRS 9).
    V0 = EAD - tarifa + comissão + custos  (ativo reconhecido inicialmente)."""
    V0 = EAD - tarifa * EAD + comissao * EAD + custos_adm * EAD
    try:
        r = brentq(lambda x: sum(PMT / (1 + x) ** t for t in range(1, prazo + 1)) - V0,
                   1e-9, 1.0, xtol=1e-12)
        return r, (1 + r) ** 12 - 1, V0
    except ValueError:
        return float("nan"), float("nan"), V0


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
  div[data-testid="stButton"] button { border-radius:10px; font-weight:600; padding:.5rem 1.4rem; }
  .stButton button[kind="primary"], button[data-testid="baseButton-primary"] {
      background:#01003c !important; border-color:#01003c !important; color:#fff !important; }
  div[data-testid="stAlert"] { border-radius:10px; }
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
    tarifa = st.number_input("Tarifa (fee cobrada, s/ EAD)", 0.0, 0.50, 0.00, 0.01,
                             help="Receita de originação cobrada uma vez sobre o valor liberado. Pode ser 0.")

# ---- cálculo (metodologia consistente — embarcada) ----
def precificar(modo_ul, base_cons, comissao_int):
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
                            base_consistente=base_cons, comissao_integral=comissao_int,
                            tarifa=tarifa)
    taxa_am = (1 + taxa_aa) ** (1 / 12) - 1 if np.isfinite(taxa_aa) else float("nan")
    # taxa efetiva contábil (IFRS 9): amortiza tarifa/comissão/custos pela TIR do fluxo
    if np.isfinite(taxa_am):
        PMT = calculate_pmt(EAD, taxa_am, prazo)
        ef_am, ef_aa, V0 = taxa_efetiva_ifrs9(EAD, PMT, prazo, tarifa, comissao, custos_adm)
    else:
        ef_am = ef_aa = float("nan")
        V0 = EAD - tarifa * EAD + comissao * EAD + custos_adm * EAD
    return dict(funding_cost=funding_cost, s_el=s_el, s_ul=s_ul, alvo=alvo, K=K,
                taxa_aa=taxa_aa, taxa_am=taxa_am, tarifa=tarifa, V0=V0,
                taxa_ef_am=ef_am, taxa_ef_aa=ef_aa)

# ---- botão calcular + resultado ----
st.subheader("Resultado")

if st.button("📊 Calcular taxa", type="primary"):
    st.session_state["calc"] = {
        "res": precificar("premio", True, True),
        "inputs": dict(ID=ID, EAD=EAD, prazo=prazo, PD=PD, LGD=LGD, di_futuro=di_futuro,
                       di_atual=di_atual, funding_pct=funding_pct, custo_capital=custo_capital,
                       pis=pis, comissao=comissao, custos_adm=custos_adm, ir_cs=ir_cs,
                       fator_pond=fator_pond, tarifa=tarifa),
    }

if "calc" not in st.session_state:
    st.caption("Ajuste os parâmetros acima e clique em **Calcular taxa**.")
else:
    res = st.session_state["calc"]["res"]
    inp = st.session_state["calc"]["inputs"]
    if not np.isfinite(res["taxa_aa"]):
        st.error("Não foi possível resolver a taxa com esses parâmetros. Revise as entradas.")
    else:
        st.success("✓ Cálculo realizado")
        st.markdown("**Taxa nominal (contrato)**")
        n1, n2 = st.columns(2)
        n1.metric("Nominal (% a.a.)", f"{res['taxa_aa']*100:.2f}%")
        n2.metric("Nominal (% a.m.)", f"{res['taxa_am']*100:.2f}%")
        st.markdown("**Taxa efetiva contábil — IFRS 9 / CMN 4.966**")
        e1, e2 = st.columns(2)
        e1.metric("Efetiva (% a.a.)", f"{res['taxa_ef_aa']*100:.2f}%" if np.isfinite(res['taxa_ef_aa']) else "—")
        e2.metric("Efetiva (% a.m.)", f"{res['taxa_ef_am']*100:.2f}%" if np.isfinite(res['taxa_ef_am']) else "—")
        st.caption("A efetiva é a TIR que amortiza tarifa (receita) e comissão/custos de originação "
                   f"ao longo da vida. Ativo reconhecido inicialmente (V₀) = R$ {res['V0']:,.2f}.")
        st.markdown("**Composição (anual)**")
        st.dataframe(pd.DataFrame({
            "Componente": ["Funding cost", "Spread EL", "Spread UL", "RAROC alvo",
                           "K_IRB (capital)", "Tarifa (receita orig.)", "Ativo reconhecido (V₀)"],
            "Valor": [f"{res['funding_cost']*100:.2f}%", f"{res['s_el']*100:.2f}%",
                      f"{res['s_ul']*100:.2f}%", f"{res['alvo']*100:.2f}%", f"R$ {res['K']:,.2f}",
                      f"R$ {res['tarifa']*inp['EAD']:,.2f}", f"R$ {res['V0']:,.2f}"],
        }), hide_index=True)

        export = pd.DataFrame([{
            "ID Client": inp["ID"], "Simulation Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Methodology": "Consistent",
            "Loan Amount (EAD)": inp["EAD"], "Term (months)": inp["prazo"], "PD": inp["PD"], "LGD": inp["LGD"],
            "Funding Rate - Duration (% a.a.)": inp["di_futuro"], "Funding Rate - Risk Free (% a.a.)": inp["di_atual"],
            "Funding Transfer Price (%)": inp["funding_pct"], "Capital Premium (p.p.)": inp["custo_capital"],
            "PIS/COFINS Tax (%)": inp["pis"], "Commission Fee (%)": inp["comissao"], "Admin Costs (%)": inp["custos_adm"],
            "IR + CS Tax (%)": inp["ir_cs"], "Risk Weight (FPR)": inp["fator_pond"],
            "Tarifa (%)": inp["tarifa"],
            "Funding Cost (%)": res["funding_cost"], "Spread EL (%)": res["s_el"],
            "Spread UL (%)": res["s_ul"], "K_IRB (R$)": res["K"],
            "Accounting Asset V0 (R$)": res["V0"],
            "Min Interest Rate (a.a.)": res["taxa_aa"], "Min Interest Rate (a.m.)": res["taxa_am"],
            "Effective Rate IFRS9 (a.a.)": res["taxa_ef_aa"], "Effective Rate IFRS9 (a.m.)": res["taxa_ef_am"],
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
