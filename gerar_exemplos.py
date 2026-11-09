"""Gera as bases de exemplo de exemplos/, cada uma isolando um diagnóstico.

Simula o que o banco OBSERVA (write-off e recuperação acumulados até a idade da
safra), dados parâmetros "verdadeiros" de PD e LGD, e grava o que foi
PRECIFICADO ao lado. A reconciliação tem que recuperar os parâmetros verdadeiros
e acusar a diferença contra o precificado.
"""

import numpy as np
import pandas as pd
import reconciliacao as R

REF = 2026 * 12 + 11          # referência: nov/2026
LAG_WO = 12                   # meses do default até a baixa
MEIA_VIDA_REC = 18            # meia-vida da recuperação de write-off


def observar(idade, prazo, taxa_am, pd_real, lgd_real, vop):
    """PEC e REC acumulados em R$ que o banco veria naquela idade da safra."""
    life = 1 - (1 - pd_real) ** (prazo / 12)
    s = R._perfil_default(prazo, taxa_am)
    haz = life / prazo
    pec = sum(haz * s[t] for t in range(prazo) if t + LAG_WO <= idade)
    rec = sum(haz * s[t] * (1 - lgd_real) * (1 - 0.5 ** ((idade - (t + LAG_WO)) / MEIA_VIDA_REC))
              for t in range(prazo) if t + LAG_WO <= idade)
    return round(pec * vop, 2), round(rec * vop, 2)


def safra(ano, mes, prazo, taxa_am, pd_real, lgd_real, pd_prec, lgd_prec, vop):
    idade = REF - (ano * 12 + mes)
    pec, rec = observar(idade, prazo, taxa_am, pd_real, lgd_real, vop)
    return dict(safra=f"{ano}-{mes:02d}", vop=vop, prazo=prazo,
                taxa_am=round(taxa_am, 6), idade_meses=idade, pec=pec, rec=rec,
                pd_precificada=pd_prec, lgd_precificada=lgd_prec)


# ---------------------------------------------------------------------------
# 01 — carteira calibrada: o modelo acertou. É o caso de controle.
# ---------------------------------------------------------------------------
def base_calibrada():
    PD, LGD, PRAZO, TAXA = 0.017, 0.75, 24, 0.021599
    linhas = []
    for ano, mes in [(2023, m) for m in range(1, 13)] + [(2024, m) for m in range(1, 13)]:
        vop = round(9_000_000 * (1 + 0.02 * ((ano - 2023) * 12 + mes)), 2)
        linhas.append(safra(ano, mes, PRAZO, TAXA, PD, LGD, PD, LGD, vop))
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 02 — PD real deteriorando, preço parado: subprecificação crescente.
# ---------------------------------------------------------------------------
def base_subprecificacao():
    PD_PREC, LGD, PRAZO, TAXA = 0.017, 0.75, 24, 0.021599
    linhas = []
    for ano, mes in [(2023, m) for m in range(1, 13)] + [(2024, m) for m in range(1, 13)]:
        k = (ano - 2023) * 12 + mes - 1
        pd_real = PD_PREC * (1 + 0.020 * k)        # PD real sobe ~2%/mês
        vop = round(9_000_000 * (1 + 0.03 * k), 2)  # e a produção acelera
        linhas.append(safra(ano, mes, PRAZO, TAXA, pd_real, LGD, PD_PREC, LGD, vop))
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 03 — LGD precificada otimista: a PD está certa, a recuperação não vem.
# ---------------------------------------------------------------------------
def base_lgd_otimista():
    PD, LGD_REAL, LGD_PREC, PRAZO, TAXA = 0.017, 0.78, 0.60, 24, 0.021599
    linhas = []
    for ano, mes in [(2023, m) for m in range(1, 13)] + [(2024, m) for m in range(1, 13)]:
        vop = round(9_000_000 * (1 + 0.02 * ((ano - 2023) * 12 + mes)), 2)
        linhas.append(safra(ano, mes, PRAZO, TAXA, PD, LGD_REAL, PD, LGD_PREC, vop))
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 04 — multiproduto: prazos de 12 a 48 meses na mesma base.
#      Mesmo risco em todos; o (PEC-REC)/VOP bruto difere MUITO entre prazos,
#      e é a duration que os traz para a mesma régua.
# ---------------------------------------------------------------------------
def base_multiproduto():
    PD, LGD = 0.017, 0.75
    produtos = [(12, 0.0235), (24, 0.021599), (36, 0.0205), (48, 0.0198)]
    linhas = []
    for prazo, taxa in produtos:
        for ano, mes in [(2023, 3), (2023, 9), (2024, 3), (2024, 9)]:
            vop = round(4_000_000 + 500_000 * (prazo / 12), 2)
            linhas.append(safra(ano, mes, prazo, taxa, PD, LGD, PD, LGD, vop))
    return pd.DataFrame(linhas).sort_values(["prazo", "safra"])


# ---------------------------------------------------------------------------
# 05 — safras jovens: mostra a marcação de imaturidade e o risco de ler cedo.
# ---------------------------------------------------------------------------
def base_safras_jovens():
    PD, LGD, PRAZO, TAXA = 0.017, 0.75, 24, 0.021599
    linhas = []
    for ano, mes in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 11)]:
        vop = round(11_000_000 * (1 + 0.015 * ((ano - 2025) * 12 + mes)), 2)
        linhas.append(safra(ano, mes, PRAZO, TAXA, PD, LGD, PD, LGD, vop))
    return pd.DataFrame(linhas)


BASES = {
    "01_carteira_calibrada.csv": (base_calibrada, "modelo acertou — gap ~0"),
    "02_subprecificacao_pd.csv": (base_subprecificacao, "PD real subiu, preço parado"),
    "03_lgd_otimista.csv": (base_lgd_otimista, "recuperação pior que a precificada"),
    "04_multiproduto.csv": (base_multiproduto, "prazos 12-48m, duration normalizando"),
    "05_safras_jovens.csv": (base_safras_jovens, "imaturas — o que NÃO usar"),
}

if __name__ == "__main__":
    import os
    os.makedirs("exemplos", exist_ok=True)
    for nome, (fn, desc) in BASES.items():
        df = fn()
        df.to_csv(f"exemplos/{nome}", index=False)
        d = R.reconciliar(df, lag_wo=LAG_WO, meia_vida_rec=MEIA_VIDA_REC)
        r = R.resumo_ponderado(d)
        n_im = int((~d["confiavel"]).sum())
        if r:
            print(f"{nome:<32} {desc:<38} realizado {r['spread_el_realizado']*100:6.3f}%  "
                  f"precificado {r['spread_el_precificado']*100:6.3f}%  "
                  f"gap {r['gap_bps']:+7.1f} bps  ({len(df)} safras, {n_im} imaturas)")
        else:
            print(f"{nome:<32} {desc:<38} nenhuma safra confiável "
                  f"({len(df)} safras, {n_im} imaturas)")
