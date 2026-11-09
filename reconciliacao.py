"""
Reconciliação de safras — perda realizada (PEC-REC)/VOP contra o EL precificado.

PEC = lançamento a write-off (baixa para prejuízo), acumulado da safra, em R$
REC = recuperação de write-off, acumulada da safra, em R$
VOP = produção da safra (desembolso), em R$

PONTE COM O PREÇO
-----------------
    spread_EL (% a.a.)  =  [(PEC - REC) / VOP]  /  duration_safra

A duration (saldo devedor médio da safra, em anos) faz dois ajustes de uma vez:
converte a base de DESEMBOLSO para EXPOSIÇÃO MÉDIA e anualiza. O resultado é
comparável, em pontos-base, ao spread de EL embutido na taxa cobrada (PD x LGD).

MATURAÇÃO (o ponto delicado)
----------------------------
A safra não fecha linearmente. O write-off satura cedo (prazo + lag de baixa),
mas a recuperação tem cauda longa. Entre os dois, o indicador ULTRAPASSA o valor
final e desce depois: com prazo 24m, lag 12m e meia-vida de recuperação 18m, ele
atinge ~118,5% do valor verdadeiro no mês 36 e só converge por volta do mês 120.

Por isso toda safra é corrigida por dois fatores de completude — um do PEC, outro
da REC — antes de virar spread. Safras abaixo de MATURIDADE_MINIMA são marcadas
como não confiáveis: o fator fica grande demais e amplifica ruído.

Os parâmetros de maturação (lag_wo, meia_vida_rec) são PREMISSAS. Calibre-os com
safras antigas reais antes de usar o módulo para decidir preço.
"""

import numpy as np
import pandas as pd

HORIZONTE_FECHAMENTO = 36      # meses: idade a partir da qual a safra é reportada
MATURIDADE_MINIMA = 18         # meses: abaixo disso o ajuste não é confiável

COLUNAS_OBRIGATORIAS = ["safra", "vop", "prazo", "taxa_am", "idade_meses", "pec", "rec"]


# ----------------------------------------------------------------------------
#  Núcleo
# ----------------------------------------------------------------------------
def duration_safra(prazo, taxa_am):
    """Saldo devedor médio de um fluxo Price, em ANOS por R$1 desembolsado.

    É o fator que converte uma perda medida sobre o desembolso em uma taxa
    anual sobre exposição — a mesma base em que o spread de EL é cotado.
    """
    if prazo <= 0 or taxa_am <= 0:
        return float("nan")
    pmt = taxa_am / (1 - (1 + taxa_am) ** -prazo)
    saldo, soma = 1.0, 0.0
    for _ in range(prazo):
        soma += saldo
        saldo = saldo * (1 + taxa_am) - pmt
    return soma / 12.0


def _perfil_default(prazo, taxa_am):
    """Saldo exposto em cada mês da safra (hazard de default uniforme)."""
    pmt = taxa_am / (1 - (1 + taxa_am) ** -prazo)
    saldo, saldos = 1.0, []
    for _ in range(prazo):
        saldos.append(saldo)
        saldo = saldo * (1 + taxa_am) - pmt
    return np.array(saldos)


def completude_pec(idade, prazo, taxa_am, lag_wo=12):
    """Fração do write-off FINAL da safra já lançada aos `idade` meses.

    Um default no mês t vira baixa no mês t + lag_wo. A fração é ponderada pelo
    saldo exposto, porque default cedo carrega exposição maior.
    """
    s = _perfil_default(prazo, taxa_am)
    total = s.sum()
    if total <= 0:
        return float("nan")
    realizado = s[np.arange(prazo) + lag_wo <= idade].sum()
    return realizado / total


def completude_rec(idade, prazo, taxa_am, lag_wo=12, meia_vida_rec=18):
    """Fração da recuperação FINAL da safra já realizada aos `idade` meses.

    Cada write-off do mês t+lag_wo recupera segundo 1 - 0,5^(tempo/meia_vida).
    A cauda é longa: é ela que faz o indicador bruto estourar o valor final.
    """
    s = _perfil_default(prazo, taxa_am)
    total = s.sum()
    if total <= 0:
        return float("nan")
    acum = 0.0
    for t in range(prazo):
        mes_wo = t + lag_wo
        if mes_wo <= idade:
            decorrido = idade - mes_wo
            acum += s[t] * (1 - 0.5 ** (decorrido / meia_vida_rec))
    return acum / total


def reconciliar(df, lag_wo=12, meia_vida_rec=18,
                horizonte=HORIZONTE_FECHAMENTO, maturidade_minima=MATURIDADE_MINIMA):
    """Recebe safras e devolve o realizado contra o precificado, em bps.

    Colunas obrigatórias:
        safra        rótulo (ex. "2024-03")
        vop          produção da safra, R$
        prazo        prazo médio contratado, meses
        taxa_am      taxa nominal média cobrada, decimal ao mês
        idade_meses  meses decorridos desde a originação
        pec          write-off bruto acumulado, R$
        rec          recuperação de write-off acumulada, R$
    Opcionais (para o lado precificado):
        pd_precificada, lgd_precificada   decimais; ou
        spread_el_precificado             decimal a.a. (tem precedência)
    """
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas ausentes: {', '.join(faltando)}")

    d = df.copy()
    d["duration"] = [duration_safra(p, t) for p, t in zip(d["prazo"], d["taxa_am"])]

    # --- observado, sem ajuste (o que a contabilidade mostra hoje) ----------
    d["pec_vop"] = d["pec"] / d["vop"]
    d["rec_vop"] = d["rec"] / d["vop"]
    d["loss_rate_bruto"] = d["pec_vop"] - d["rec_vop"]
    d["spread_el_bruto"] = d["loss_rate_bruto"] / d["duration"]

    # --- ajuste de maturação, separado por componente -----------------------
    d["compl_pec"] = [completude_pec(i, p, t, lag_wo)
                      for i, p, t in zip(d["idade_meses"], d["prazo"], d["taxa_am"])]
    d["compl_rec"] = [completude_rec(i, p, t, lag_wo, meia_vida_rec)
                      for i, p, t in zip(d["idade_meses"], d["prazo"], d["taxa_am"])]

    d["pec_final"] = np.where(d["compl_pec"] > 0, d["pec_vop"] / d["compl_pec"], np.nan)
    d["rec_final"] = np.where(d["compl_rec"] > 0, d["rec_vop"] / d["compl_rec"], np.nan)
    d["loss_rate_ajustado"] = d["pec_final"] - d["rec_final"]
    d["spread_el_realizado"] = d["loss_rate_ajustado"] / d["duration"]

    # --- lado precificado ---------------------------------------------------
    if "spread_el_precificado" in d.columns:
        d["spread_el_precificado"] = d["spread_el_precificado"].astype(float)
    elif {"pd_precificada", "lgd_precificada"}.issubset(d.columns):
        d["spread_el_precificado"] = d["pd_precificada"] * d["lgd_precificada"]
    else:
        d["spread_el_precificado"] = np.nan

    d["gap_bps"] = (d["spread_el_realizado"] - d["spread_el_precificado"]) * 1e4

    # --- confiabilidade -----------------------------------------------------
    def _status(idade):
        if idade >= horizonte:
            return "Fechada"
        if idade >= maturidade_minima:
            return "Em maturação"
        return "Imatura — não usar"
    d["status"] = d["idade_meses"].apply(_status)
    d["confiavel"] = d["idade_meses"] >= maturidade_minima
    return d


def resumo_ponderado(d):
    """Agrega só as safras confiáveis, ponderando pela produção."""
    c = d[d["confiavel"] & d["spread_el_realizado"].notna()]
    if c.empty:
        return None
    w = c["vop"]
    real = float((c["spread_el_realizado"] * w).sum() / w.sum())
    prec = c["spread_el_precificado"]
    prec_p = float((prec * w).sum() / w.sum()) if prec.notna().all() else float("nan")
    return {"n_safras": int(len(c)), "vop": float(w.sum()),
            "spread_el_realizado": real, "spread_el_precificado": prec_p,
            "gap_bps": (real - prec_p) * 1e4}


# ----------------------------------------------------------------------------
#  Aba Streamlit (plugar no app com: from reconciliacao import render_reconciliacao)
# ----------------------------------------------------------------------------
def render_reconciliacao(st):
    st.markdown("### Reconciliação de safras — EL realizado × EL precificado")
    st.caption("Compara a perda efetivamente realizada por baixa, safra a safra, com o "
               "spread de EL que foi embutido no preço. A ponte é "
               "**(PEC − REC)/VOP ÷ duration da safra**.")

    with st.expander("Premissas de maturação — calibre com o seu histórico"):
        m1, m2, m3 = st.columns(3)
        lag_wo = m1.number_input("Lag até o write-off (meses)", 0, 60, 12, 1,
                                 help="Meses entre o default e a baixa para prejuízo.")
        meia_vida = m2.number_input("Meia-vida da recuperação (meses)", 1, 120, 18, 1,
                                    help="Tempo para recuperar metade do que ainda virá.")
        horizonte = m3.number_input("Horizonte de fechamento (meses)", 12, 120,
                                    HORIZONTE_FECHAMENTO, 6)
        st.caption("Estes são os dois números que governam o ajuste. São premissas, não "
                   "dados: no mês 36 o write-off já saturou mas a recuperação não, e o "
                   "indicador bruto chega a ~118% do valor verdadeiro. Calibre com safras "
                   "antigas antes de decidir preço com isto.")

    up = st.file_uploader("Safras (CSV)", type=["csv"], key="recon_csv")
    st.caption("Colunas: safra, vop, prazo, taxa_am, idade_meses, pec, rec "
               "— e pd_precificada + lgd_precificada (ou spread_el_precificado).")

    if up is None:
        st.info("Envie o CSV das safras para rodar a reconciliação.")
        return

    try:
        bruto = pd.read_csv(up)
        d = reconciliar(bruto, lag_wo=lag_wo, meia_vida_rec=meia_vida, horizonte=horizonte)
    except Exception as e:
        st.error(f"Não foi possível reconciliar: {e}")
        return

    r = resumo_ponderado(d)
    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("EL realizado (pond.)", f"{r['spread_el_realizado']*100:.3f}% a.a.")
        c2.metric("EL precificado (pond.)", f"{r['spread_el_precificado']*100:.3f}% a.a.")
        c3.metric("Gap", f"{r['gap_bps']:+.1f} bps")
        st.caption(f"{r['n_safras']} safras confiáveis, R$ {r['vop']:,.2f} de produção. "
                   "Gap positivo = o preço cobrou MENOS EL do que a carteira entregou "
                   "(subprecificação).")

    pct = lambda x: "—" if pd.isna(x) else f"{x*100:.3f}%"
    st.dataframe(pd.DataFrame({
        "Safra": d["safra"],
        "Idade": d["idade_meses"].astype(int).astype(str) + "m",
        "Status": d["status"],
        "VOP": d["vop"].map(lambda v: f"R$ {v:,.2f}"),
        "Duration": d["duration"].map(lambda v: f"{v:.3f} a"),
        "(PEC−REC)/VOP bruto": d["loss_rate_bruto"].map(pct),
        "(PEC−REC)/VOP ajust.": d["loss_rate_ajustado"].map(pct),
        "EL realizado a.a.": d["spread_el_realizado"].map(pct),
        "EL precificado a.a.": d["spread_el_precificado"].map(pct),
        "Gap (bps)": d["gap_bps"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}"),
    }), hide_index=True, width="stretch")

    conf = d[d["confiavel"] & d["spread_el_realizado"].notna()]
    if not conf.empty:
        st.markdown("**EL realizado × precificado, por safra**")
        st.line_chart(pd.DataFrame({
            "Realizado": conf["spread_el_realizado"].values * 100,
            "Precificado": conf["spread_el_precificado"].values * 100,
        }, index=conf["safra"].values))
        st.caption("Safras imaturas ficam fora do gráfico e do agregado.")

    with st.expander("Decomposição — qual parâmetro está errado"):
        st.markdown(
"""O indicador composto só valida o **produto** PD × LGD. Separado, ele calibra os dois
parâmetros do modelo de forma independente:

- **PEC/VOP** (ajustado) → frequência de perda bruta: calibra **PD × EAD no default**
- **REC/PEC** (ajustado) → taxa de recuperação: calibra **(1 − LGD)** direto

Se o EL agregado fecha mas os dois componentes estão errados em sentidos opostos, os
erros se cancelaram por acaso — e deixarão de se cancelar na próxima mudança de mix.""")
        st.dataframe(pd.DataFrame({
            "Safra": d["safra"],
            "PEC/VOP (ajust.)": d["pec_final"].map(pct),
            "REC/PEC (ajust.)": (d["rec_final"] / d["pec_final"]).map(pct),
            "LGD implícita": (1 - d["rec_final"] / d["pec_final"]).map(pct),
            "LGD precificada": (d["lgd_precificada"].map(pct)
                                if "lgd_precificada" in d.columns else "—"),
        }), hide_index=True, width="stretch")
