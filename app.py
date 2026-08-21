"""
Interest Rate Loan Pricing — RAROC (Consignado INSS)
Porte do app Shiny para Streamlit.  Modelo: Walter Correa Neto (BFEng).

Roda de graça no Streamlit Community Cloud.  Arquivo único, sem banco de dados.

Metodologia embarcada = "Consistente" (correções de validação):
  (1) spread UL apenas sobre o prêmio de capital;
  (2) base temporal anual = anual (sem o fator n/12 no alvo);
  (3) comissão/custos integrais (one-time), sem o fator 12/n;
  (4) V0 da TJEO capitaliza apenas custos incrementais e diretamente atribuiveis
      (CMN 4.966, art. 2, XXII); custos de estrutura ficam fora do valor contabil
      bruto e entram somente na precificacao (RAROC).
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


def calcular_tjeo(base, PMT, prazo, tarifa_rs, comissao_rs, custos_orig_rs):
    """TJEO — taxa de juros efetiva original (CMN 4.966, art. 2º, XXII).

    Taxa que equaliza o valor presente de todos os recebimentos e pagamentos ao
    longo do prazo contratual ao VALOR CONTÁBIL BRUTO do ativo no reconhecimento
    inicial:  V₀ = base − tarifa recebida + comissão paga + custos de originação.

    Só entram custos INCREMENTAIS e DIRETAMENTE ATRIBUÍVEIS à originação
    (comissão ao correspondente, averbação, registro de garantia, análise
    terceirizada).  Despesas administrativas de estrutura NÃO são capitalizáveis
    e não entram aqui — são insumo de precificação (RAROC), não de mensuração.
    """
    V0 = base - tarifa_rs + comissao_rs + custos_orig_rs
    try:
        r = brentq(lambda x: sum(PMT / (1 + x) ** t for t in range(1, prazo + 1)) - V0,
                   1e-9, 1.0, xtol=1e-12)
        return r, (1 + r) ** 12 - 1, V0
    except ValueError:
        return float("nan"), float("nan"), V0


def iof_pf(valor, prazo_meses, aliq_adicional=0.0038, aliq_diaria=0.000082):
    """IOF pessoa física: adicional fixo + diário (limitado a 365 dias)."""
    dias = min(prazo_meses * 30, 365)
    return valor * aliq_adicional + valor * aliq_diaria * dias


def montar_operacao(EAD_sol, prazo, tarifa, seguro, incluir_iof,
                    iof_add, iof_dia, regime="financiado"):
    """Monta a operação a partir do valor pedido pelo cliente.

    regime="financiado" (padrão, consignado): o cliente recebe o valor que pediu e
      IOF + tarifa + seguro entram no valor financiado (pagos diluídos na parcela).
      Como o IOF incide sobre o próprio financiado, há circularidade — resolvida em
      forma fechada:  F = (EAD + tarifa + seguro) / (1 - k),  k = alíquota total.
    regime="retido": encargos descontados da liberação (financiado = EAD pedido).

    Retorna: financiado (EAD de risco), IOF R$, líquido liberado, tarifa R$, seguro R$.
    """
    tarifa_rs = tarifa * EAD_sol
    seguro_rs = seguro * EAD_sol
    if regime == "financiado":
        dias = min(prazo * 30, 365)
        k = (iof_add + iof_dia * dias) if incluir_iof else 0.0
        financiado = (EAD_sol + tarifa_rs + seguro_rs) / (1 - k)
        iof_val = financiado * k
        liquido = EAD_sol                       # recebe cheio o que pediu
    else:
        iof_val = iof_pf(EAD_sol, prazo, iof_add, iof_dia) if incluir_iof else 0.0
        financiado = EAD_sol
        liquido = EAD_sol - tarifa_rs - seguro_rs - iof_val
    return financiado, iof_val, liquido, tarifa_rs, seguro_rs


def calcular_cet(PMT, prazo, liquido):
    """CET (custo efetivo total, olhar do cliente): TIR que iguala o valor líquido
    efetivamente recebido ao fluxo de parcelas que o cliente paga."""
    if liquido <= 0:
        return float("nan"), float("nan")
    try:
        r = brentq(lambda x: sum(PMT / (1 + x) ** t for t in range(1, prazo + 1)) - liquido,
                   1e-9, 2.0, xtol=1e-12)
        return r, (1 + r) ** 12 - 1
    except ValueError:
        return float("nan"), float("nan")


def _tir(PMT, prazo, base):
    """TIR mensal de um fluxo Price: PMT constante descontado contra `base`."""
    if not np.isfinite(base) or base <= 0 or not np.isfinite(PMT):
        return float("nan")
    try:
        return brentq(lambda x: sum(PMT / (1 + x) ** t for t in range(1, prazo + 1)) - base,
                      1e-9, 2.0, xtol=1e-12)
    except ValueError:
        return float("nan")


def decompor_taxa(PMT, prazo, financiado, iof_val, tarifa_rs, seguro_rs):
    """Decomposição exata da taxa por TIR incremental (cascata nominal → CET).

    Mantém a PARCELA fixa e vai retirando da base, uma camada por vez, o que o
    cliente não recebeu.  A diferença entre TIRs sucessivas é o impacto, em p.p.,
    daquele encargo.  A ordem segue a convenção do CET: juro, tributo (IOF),
    tarifas e, por último, seguros — cada camada responde "quanto ESTE item
    acrescenta, dado tudo que já veio antes".  A soma fecha no CET por construção.
    """
    camadas = [("IOF (tributo)", iof_val), ("Tarifa de cadastro", tarifa_rs),
               ("Seguro", seguro_rs)]
    base = financiado
    taxa_ant = _tir(PMT, prazo, base)
    linhas = [{"Camada": "Juro (taxa nominal do contrato)", "Base (PV)": base,
               "Taxa a.m.": taxa_ant, "Impacto (p.p. a.m.)": float("nan")}]
    for nome, valor in camadas:
        if valor <= 0:
            continue
        base -= valor
        taxa = _tir(PMT, prazo, base)
        linhas.append({"Camada": f"+ {nome}", "Base (PV)": base, "Taxa a.m.": taxa,
                       "Impacto (p.p. a.m.)": taxa - taxa_ant})
        taxa_ant = taxa
    linhas.append({"Camada": "= CET (custo total ao cliente)", "Base (PV)": base,
                   "Taxa a.m.": taxa_ant, "Impacto (p.p. a.m.)": float("nan")})
    return linhas


def brl(x):
    """Formata em reais no padrão pt-BR (R$ 1.234,56)."""
    if not np.isfinite(x):
        return "—"
    return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x):
    """Formata taxa em % pt-BR (2,16%)."""
    if not np.isfinite(x):
        return "—"
    return f"{x*100:.2f}".replace(".", ",") + "%"


def gerar_demonstrativo(ID, res, EAD, prazo, seguro):
    """Documento voltado ao CLIENTE — apenas o que o banco pode/deve divulgar.
    NÃO inclui funding, spreads, RAROC, K_IRB, comissão, PD/LGD ou taxa efetiva IFRS 9."""
    tarifa_rs = res["tarifa_rs"]
    seguro_rs = res["seguro_rs"]
    total = res["parcela"] * prazo if np.isfinite(res["parcela"]) else float("nan")
    L = []
    L.append("DEMONSTRATIVO DE CUSTO EFETIVO TOTAL (CET)")
    L.append("Operação de Crédito")
    L.append("=" * 52)
    L.append(f"Cliente (ID) : {ID or '-'}")
    L.append(f"Data         : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    L.append("")
    L.append("VALORES")
    L.append(f"  Valor solicitado ................. {brl(EAD)}")
    L.append(f"  IOF .............................. {brl(res['iof_val'])}")
    L.append(f"  Tarifa de cadastro ............... {brl(tarifa_rs)}")
    L.append(f"  Seguro ........................... {brl(seguro_rs)}")
    L.append(f"  Valor total financiado ........... {brl(res['financiado'])}")
    L.append(f"  Valor líquido liberado ao cliente  {brl(res['liquido'])}")
    if res.get("regime") == "financiado":
        L.append("  (IOF, tarifa e seguro financiados e diluídos nas parcelas)")
    L.append("")
    L.append("CONDIÇÕES DE PAGAMENTO")
    L.append(f"  Prazo ............................ {prazo} meses")
    L.append(f"  Sistema de amortização ........... Price (parcelas fixas)")
    L.append(f"  Valor da parcela ................. {brl(res['parcela'])}")
    L.append(f"  Total a pagar .................... {brl(total)}")
    L.append("")
    L.append("TAXAS")
    L.append(f"  Taxa de juros .................... {pct(res['taxa_am'])} a.m.  |  {pct(res['taxa_aa'])} a.a.")
    L.append(f"  Custo Efetivo Total (CET) ........ {pct(res['cet_am'])} a.m.  |  {pct(res['cet_aa'])} a.a.")
    L.append("")
    L.append("-" * 52)
    L.append("O Custo Efetivo Total (CET) representa o custo total da")
    L.append("operação, expresso em taxa percentual, conforme a")
    L.append("Resolução CMN nº 4.881. Inclui juros, tributos (IOF),")
    L.append("tarifas e demais encargos incidentes sobre o crédito.")
    return "\n".join(L)


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
    comissao = st.number_input("Comissão CORBAN", 0.0, 0.50, 0.06, 0.01,
                               help="Incide sobre o valor LIBERADO ao cliente, não sobre o financiado: "
                                    "não se paga comissão sobre IOF, que é tributo apenas repassado.")
with c3:
    di_atual = st.number_input("Funding Rate – Risk Free (% a.a.)", 0.1, 50.0, 11.70, 0.01)
    funding_pct = st.number_input("Funding Transfer Price Factor (%)", 100.0, 500.0, 128.0, 10.0)
    custo_capital = st.number_input("Capital Premium (p.p.)", 0.0, 15.0, 5.0, 0.5)
    custos_orig = st.number_input("Custos de originação — capitalizáveis", 0.0, 0.50, 0.00, 0.005,
                                  format="%.3f",
                                  help="Custos INCREMENTAIS e diretamente atribuíveis à originação "
                                       "(averbação, registro, análise terceirizada). Entram no V₀ "
                                       "da TJEO e no RAROC.")
with c4:
    fator_pond = st.number_input("Risk Weighting (FPR)", 0.0, 0.99, 0.50, 0.01)
    ir_cs = st.number_input("IR/CS Tax", 0.0, 0.80, 0.40, 0.01)
    tarifa = st.number_input("Tarifa (fee cobrada, s/ EAD)", 0.0, 0.50, 0.00, 0.01,
                             help="Receita de originação cobrada uma vez sobre o valor liberado. Pode ser 0.")
    custos_estrut = st.number_input("Custos administrativos — estrutura", 0.0, 0.50, 0.01, 0.005,
                                    format="%.3f",
                                    help="Rateio de estrutura/overhead. NÃO é capitalizável sob a "
                                         "CMN 4.966 — entra só no RAROC, fora do V₀ da TJEO.")
custos_adm = custos_orig + custos_estrut     # despesa total: base do RAROC

with st.expander("Encargos ao cliente — regime, IOF e seguro"):
    regime = st.radio(
        "Regime dos encargos (IOF, tarifa e seguro)",
        ["Financiados (cliente recebe o valor pedido)", "Retidos (descontados da liberação)"],
        index=0, horizontal=True,
        help="No regime financiado o cliente recebe o valor que pediu e os encargos entram "
             "no valor financiado — logo, também na exposição de risco e na parcela.")
    regime_key = "financiado" if regime.startswith("Financiados") else "retido"
    ec1, ec2, ec3 = st.columns(3)
    seguro = ec1.number_input("Seguro (s/ EAD)", 0.0, 0.50, 0.00, 0.01,
                              help="Prêmio de seguro retido na originação. Pode ser 0.")
    incluir_iof = ec1.checkbox("Incluir IOF (pessoa física)", value=True)
    iof_add = ec2.number_input("IOF adicional — fixo", 0.0, 0.05, 0.0038, 0.0001, format="%.4f",
                               help="Alíquota fixa do IOF (padrão PF ≈ 0,38%).")
    iof_dia = ec3.number_input("IOF diário — por dia (máx 365d)", 0.0, 0.001, 0.000082, 0.000001,
                               format="%.6f", help="Alíquota diária do IOF (padrão PF ≈ 0,0082%/dia).")

# ---- cálculo (metodologia consistente — embarcada) ----
def precificar(modo_ul, base_cons, comissao_int):
    # 1) monta a operação: o EAD digitado é o valor PEDIDO pelo cliente
    F, iof_val, liquido, tarifa_rs, seguro_rs = montar_operacao(
        EAD, prazo, tarifa, seguro, incluir_iof, iof_add, iof_dia, regime_key)

    # 2) risco e precificação rodam sobre o valor FINANCIADO (a exposição do banco)
    tarifa_ratio = tarifa_rs / F                  # tarifa em R$ fixo, expressa s/ o financiado
    # comissão ao correspondente incide sobre o VALOR LIBERADO ao cliente: não se paga
    # comissão sobre IOF (tributo apenas repassado à Receita, não é crédito originado)
    comissao_rs = comissao * liquido
    comissao_ratio = comissao_rs / F              # idem, expressa s/ o financiado p/ o RAROC
    rho = calcular_rho(PD, prazo)
    funding_cost = (di_futuro / 100) * (funding_pct / 100)
    premio = custo_capital / 100
    s_el = spread_el(PD, LGD, prazo, funding_cost)
    s_ul = spread_ul(F, LGD, PD, prazo, funding_cost, rho, premio,
                     di_futuro / 100, di_atual / 100, modo=modo_ul)
    alvo = funding_cost + s_el + s_ul
    K = calculate_IRB_capital(F, LGD, PD, rho, prazo)
    taxa_aa = otimizar_taxa(F, LGD, PD, rho, funding_cost, prazo, alvo,
                            pis, comissao_ratio, custos_adm, ir_cs, fator_pond, di_atual / 100,
                            base_consistente=base_cons, comissao_integral=comissao_int,
                            tarifa=tarifa_ratio)
    taxa_am = (1 + taxa_aa) ** (1 / 12) - 1 if np.isfinite(taxa_aa) else float("nan")
    if np.isfinite(taxa_am):
        # 3) parcela sobre o financiado (IOF/tarifa/seguro diluídos, no regime financiado)
        PMT = calculate_pmt(F, taxa_am, prazo)
        # TJEO (CMN 4.966): só custos capitalizáveis entram no valor contábil bruto
        ef_am, ef_aa, V0 = calcular_tjeo(F, PMT, prazo, tarifa_rs,
                                         comissao_rs, custos_orig * F)
        # CET (olhar do cliente): TIR das parcelas contra o que ele recebeu de fato
        cet_am, cet_aa = calcular_cet(PMT, prazo, liquido)
    else:
        ef_am = ef_aa = cet_am = cet_aa = float("nan")
        V0 = F - tarifa_rs + comissao_rs + custos_orig * F
        PMT = float("nan")
    return dict(funding_cost=funding_cost, s_el=s_el, s_ul=s_ul, alvo=alvo, K=K,
                taxa_aa=taxa_aa, taxa_am=taxa_am, tarifa=tarifa, V0=V0,
                custos_orig_rs=custos_orig * F, custos_estrut_rs=custos_estrut * F,
                comissao_rs=comissao_rs,
                financiado=F, tarifa_rs=tarifa_rs, seguro_rs=seguro_rs, regime=regime_key,
                taxa_ef_am=ef_am, taxa_ef_aa=ef_aa,
                cet_am=cet_am, cet_aa=cet_aa, iof_val=iof_val, liquido=liquido,
                parcela=PMT)

# ---- botão calcular + resultado ----
st.subheader("Resultado")

if st.button("📊 Calcular taxa", type="primary"):
    st.session_state["calc"] = {
        "res": precificar("premio", True, True),
        "inputs": dict(ID=ID, EAD=EAD, prazo=prazo, PD=PD, LGD=LGD, di_futuro=di_futuro,
                       di_atual=di_atual, funding_pct=funding_pct, custo_capital=custo_capital,
                       pis=pis, comissao=comissao, custos_orig=custos_orig,
                       custos_estrut=custos_estrut, ir_cs=ir_cs,
                       fator_pond=fator_pond, tarifa=tarifa,
                       seguro=seguro, incluir_iof=incluir_iof, iof_add=iof_add, iof_dia=iof_dia,
                       regime=regime_key),
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
        if res["regime"] == "financiado":
            st.info(f"**Encargos financiados** — o cliente recebe {brl(res['liquido'])} "
                    f"(o valor pedido) e financia {brl(res['financiado'])} = valor pedido "
                    f"+ IOF {brl(res['iof_val'])} + tarifa {brl(res['tarifa_rs'])}"
                    + (f" + seguro {brl(res['seguro_rs'])}" if res["seguro_rs"] > 0 else "")
                    + ". A exposição de risco (EL, K_IRB) e a parcela usam o valor financiado.")
        else:
            st.info(f"**Encargos retidos** — o cliente financia {brl(res['financiado'])} e recebe "
                    f"{brl(res['liquido'])} líquidos, já descontados IOF, tarifa e seguro.")
        st.markdown("**Taxa nominal (contrato)**")
        n1, n2 = st.columns(2)
        n1.metric("Nominal (% a.a.)", f"{res['taxa_aa']*100:.2f}%")
        n2.metric("Nominal (% a.m.)", f"{res['taxa_am']*100:.2f}%")
        st.markdown("**TJEO — Taxa de Juros Efetiva Original (CMN 4.966)**")
        e1, e2 = st.columns(2)
        e1.metric("TJEO (% a.a.)", f"{res['taxa_ef_aa']*100:.2f}%" if np.isfinite(res['taxa_ef_aa']) else "—")
        e2.metric("TJEO (% a.m.)", f"{res['taxa_ef_am']*100:.2f}%" if np.isfinite(res['taxa_ef_am']) else "—")
        st.caption("Taxa que equaliza o valor presente dos recebimentos e pagamentos do contrato ao "
                   f"**valor contábil bruto** no reconhecimento inicial: V₀ = {brl(res['V0'])} = "
                   f"financiado {brl(res['financiado'])} − tarifa {brl(res['tarifa_rs'])} + comissão "
                   f"{brl(res['comissao_rs'])} + custos de originação {brl(res['custos_orig_rs'])}. "
                   f"Custos de estrutura ({brl(res['custos_estrut_rs'])}) ficam de fora — não são "
                   "capitalizáveis.")

        with st.expander("ℹ️ Por que a taxa nominal difere da TJEO?"):
            st.markdown(
"""A taxa **nominal** e a **TJEO** usam o **mesmo fluxo de parcelas** — o que muda é a **base** contra a qual esse fluxo é medido, não o regime de tempo:

- **Nominal** = TIR das parcelas contra o **valor de face** (valor financiado).
- **TJEO** = TIR das mesmas parcelas contra o **valor contábil bruto** inicial: **V₀ = financiado − tarifa + comissão + custos de originação**.

A comissão e os custos de originação (despesas) *aumentam* V₀ → a TJEO fica **abaixo** da nominal. A tarifa (receita) *reduz* V₀ → a TJEO **sobe**. A diferença é o **efeito líquido dos ajustes de originação** embutidos em V₀ — e não uma questão de caixa vs. competência.

**O que entra em V₀ é matéria de classificação, não de calibragem.** Só custos **incrementais e diretamente atribuíveis** são capitalizáveis (comissão ao correspondente, averbação, registro, análise terceirizada). Rateio de estrutura e overhead **não** entram — permanecem como despesa do período e afetam apenas a precificação (RAROC). Incluí-los indevidamente infla o ativo e **subestima** a TJEO de forma sistemática.

Quando os ajustes de originação se anulam em V₀, a TJEO **converge** para a nominal. Como a tarifa e a comissão incidem sobre a **mesma base** — o valor liberado ao cliente —, percentuais iguais se cancelam exatamente e as duas taxas ficam idênticas (desde que não haja custos capitalizáveis). A comissão não incide sobre o valor financiado porque não se paga correspondente sobre IOF, que é tributo repassado à Receita, não crédito originado.

O efeito é **inversamente proporcional ao prazo**: o mesmo custo de originação, diluído em poucas parcelas, desloca a taxa mensal muito mais do que em prazos longos.""")
            st.caption(f"Nesta operação: nominal {res['taxa_am']*100:.3f}% a.m. · TJEO "
                       f"{res['taxa_ef_am']*100:.3f}% a.m. · V₀ = {brl(res['V0'])} · "
                       f"diferença ≈ {(res['taxa_am']-res['taxa_ef_am'])*1e4:.0f} bps.")
            st.table(pd.DataFrame({
                "Prazo (custo de originação = 2% do principal)": ["2 meses", "6 meses", "12 meses",
                                                                   "24 meses", "48 meses"],
                "Deslocamento da TJEO": ["−137 bps a.m.", "−60 bps a.m.", "−34 bps a.m.",
                                          "−19 bps a.m.", "−12 bps a.m."],
            }))
            st.caption("Tabela ilustrativa do efeito-prazo: um custo de originação de 2% do "
                       "principal derruba a TJEO em 137 bps num contrato de 2 meses e em apenas "
                       "12 bps num de 48 — por isso o ajuste nunca é 'ligeiro' por definição.")

        st.markdown("**CET — Custo Efetivo Total (olhar do cliente)**")
        t1, t2 = st.columns(2)
        t1.metric("CET (% a.a.)", f"{res['cet_aa']*100:.2f}%" if np.isfinite(res['cet_aa']) else "—")
        t2.metric("CET (% a.m.)", f"{res['cet_am']*100:.2f}%" if np.isfinite(res['cet_am']) else "—")
        if res["regime"] == "financiado":
            st.caption(f"CET = TIR das parcelas (calculadas sobre o financiado, "
                       f"{brl(res['financiado'])}) contra o que o cliente recebeu de fato "
                       f"({brl(res['liquido'])}). O IOF de {brl(res['iof_val'])} e a tarifa "
                       f"estão embutidos na parcela — por isso o CET fica acima da taxa nominal.")
        else:
            st.caption(f"CET = TIR sobre o valor líquido recebido pelo cliente "
                       f"({brl(res['liquido'])}) = EAD − tarifa − IOF ({brl(res['iof_val'])}) − seguro.")
        with st.expander("🔎 Decomposição da taxa — de onde vem cada ponto do CET"):
            linhas = decompor_taxa(res["parcela"], inp["prazo"], res["financiado"],
                                   res["iof_val"], res["tarifa_rs"], res["seguro_rs"])
            st.markdown(
"""Mesma **parcela**, bases diferentes. Cada camada retira da base o que o cliente **não recebeu**, e a variação da TIR é o custo daquele encargo. A ordem segue a convenção do CET — juro, tributo, tarifas e seguros — de modo que cada linha responde *"quanto ESTE item acrescenta, dado tudo que já veio antes"*. **A soma dos impactos fecha exatamente no CET** (decomposição exata, não aproximação).""")
            st.table(pd.DataFrame([{
                "Camada": L["Camada"],
                "Base (PV)": brl(L["Base (PV)"]),
                "Taxa a.m.": f"{L['Taxa a.m.']*100:.3f}%" if np.isfinite(L["Taxa a.m."]) else "—",
                "Impacto": (f"{L['Impacto (p.p. a.m.)']*100:+.3f} p.p."
                            if np.isfinite(L["Impacto (p.p. a.m.)"]) else "—"),
            } for L in linhas]))
            st.caption(f"Verificação: nominal {res['taxa_am']*100:.3f}% + soma dos impactos "
                       f"{(res['cet_am']-res['taxa_am'])*100:+.3f} p.p. = CET {res['cet_am']*100:.3f}% a.m. "
                       "A decomposição só é possível porque o app parte dos parâmetros de cada "
                       "encargo — principal e parcela sozinhos entregam apenas a taxa cheia.")

        st.markdown("**Composição (anual)**")
        st.dataframe(pd.DataFrame({
            "Componente": ["Valor pedido (liberado)", "Valor financiado (EAD de risco)",
                           "Funding cost", "Spread EL", "Spread UL", "RAROC alvo",
                           "K_IRB (capital)", "Tarifa (receita orig.)", "Comissão (s/ valor liberado)",
                           "Custos de originação (capitalizáveis)",
                           "Custos de estrutura (não capitalizáveis)",
                           "Valor contábil bruto (V₀)"],
            "Valor": [brl(res['liquido']), brl(res['financiado']),
                      f"{res['funding_cost']*100:.2f}%", f"{res['s_el']*100:.2f}%",
                      f"{res['s_ul']*100:.2f}%", f"{res['alvo']*100:.2f}%", brl(res['K']),
                      brl(res['tarifa_rs']), brl(res['comissao_rs']),
                      brl(res['custos_orig_rs']), brl(res['custos_estrut_rs']), brl(res['V0'])],
        }), hide_index=True)

        export = pd.DataFrame([{
            "ID Client": inp["ID"], "Simulation Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Methodology": "Consistent",
            "Requested Amount (R$)": inp["EAD"], "Charges Regime": inp["regime"],
            "Financed Amount / Risk EAD (R$)": res["financiado"],
            "Term (months)": inp["prazo"], "PD": inp["PD"], "LGD": inp["LGD"],
            "Funding Rate - Duration (% a.a.)": inp["di_futuro"], "Funding Rate - Risk Free (% a.a.)": inp["di_atual"],
            "Funding Transfer Price (%)": inp["funding_pct"], "Capital Premium (p.p.)": inp["custo_capital"],
            "PIS/COFINS Tax (%)": inp["pis"], "Commission on Disbursed Amount (%)": inp["comissao"], 
            "IR + CS Tax (%)": inp["ir_cs"], "Risk Weight (FPR)": inp["fator_pond"],
            "Origination Costs - capitalizable (%)": inp["custos_orig"],
            "Structural Admin Costs - not capitalizable (%)": inp["custos_estrut"],
            "Tarifa (%)": inp["tarifa"], "Seguro (%)": inp["seguro"],
            "IOF incluído": inp["incluir_iof"], "IOF (R$)": res["iof_val"],
            "Funding Cost (%)": res["funding_cost"], "Spread EL (%)": res["s_el"],
            "Spread UL (%)": res["s_ul"], "K_IRB (R$)": res["K"],
            "Gross Carrying Amount V0 (R$)": res["V0"], "Net Amount to Client (R$)": res["liquido"],
            "Min Interest Rate (a.a.)": res["taxa_aa"], "Min Interest Rate (a.m.)": res["taxa_am"],
            "TJEO - Original Effective Rate (a.a.)": res["taxa_ef_aa"],
            "TJEO - Original Effective Rate (a.m.)": res["taxa_ef_am"],
            "CET (a.a.)": res["cet_aa"], "CET (a.m.)": res["cet_am"],
            "Author": "WalterCN - Banking Financial Engineering - BFEng",
        }])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            export.to_excel(w, index=False, sheet_name="Simulation")
        st.download_button("⬇️ Baixar Excel", buf.getvalue(),
                           file_name=f"Simulation_{datetime.now():%Y-%m-%d}_BFEng.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ---- Documento para o cliente (apenas dados divulgáveis) ----
        st.divider()
        st.subheader("📄 Demonstrativo ao cliente (contrato)")
        st.caption("Contém apenas o que o banco pode/deve compartilhar com o cliente — "
                   "sem funding, spreads, RAROC, capital, comissão ou TJEO.")
        cliente = pd.DataFrame({
            "Item": ["Valor solicitado", "IOF", "Tarifa de cadastro", "Seguro",
                     "Valor total financiado", "Valor líquido liberado", "Prazo",
                     "Valor da parcela", "Total a pagar",
                     "Taxa de juros", "Custo Efetivo Total (CET)"],
            "Valor": [brl(inp["EAD"]), brl(res["iof_val"]), brl(res["tarifa_rs"]),
                      brl(res["seguro_rs"]), brl(res["financiado"]), brl(res["liquido"]),
                      f"{inp['prazo']} meses",
                      brl(res["parcela"]), brl(res["parcela"]*inp["prazo"]),
                      f"{pct(res['taxa_am'])} a.m. | {pct(res['taxa_aa'])} a.a.",
                      f"{pct(res['cet_am'])} a.m. | {pct(res['cet_aa'])} a.a."],
        })
        st.table(cliente)
        doc = gerar_demonstrativo(inp["ID"], res, inp["EAD"], inp["prazo"], inp["seguro"])
        st.download_button("⬇️ Baixar demonstrativo do cliente (.txt)", doc.encode("utf-8"),
                           file_name=f"Demonstrativo_CET_{datetime.now():%Y-%m-%d}.txt",
                           mime="text/plain")

st.markdown(
    '<div class="foot">paper: '
    '<a href="https://rpubs.com/WalterCN/RarocPricing" target="_blank">Link</a> · '
    'contato: <a href="mailto:walter.correa.neto@gmail.com">walter.correa.neto@gmail.com</a></div>',
    unsafe_allow_html=True)
