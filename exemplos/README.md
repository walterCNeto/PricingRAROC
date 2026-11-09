# Bases de exemplo — reconciliação de safras

Cada base isola **um** diagnóstico. Rode na aba *Reconciliação de safras* do app,
compare com o resultado esperado abaixo e você sabe ler a ferramenta.

Todas usam as premissas de maturação padrão: **lag até o write-off 12m** e
**meia-vida da recuperação 18m**. Mexer nelas muda os números.

Regeneráveis com `python gerar_exemplos.py` a partir da raiz do projeto.

---

## O layout de entrada

| coluna | o que é |
|---|---|
| `safra` | rótulo da safra, `AAAA-MM` |
| `vop` | produção da safra (desembolso), R$ |
| `prazo` | prazo médio contratado, meses |
| `taxa_am` | taxa nominal média cobrada, decimal ao mês (`0.021599` = 2,1599%) |
| `idade_meses` | meses decorridos desde a originação |
| `pec` | write-off bruto acumulado da safra, R$ |
| `rec` | recuperação de write-off acumulada da safra, R$ |
| `pd_precificada` | PD anual que entrou no preço, decimal |
| `lgd_precificada` | LGD que entrou no preço, decimal |

No lugar de `pd_precificada` + `lgd_precificada` pode vir `spread_el_precificado`
(decimal a.a.), que tem precedência.

**A ponte com o preço:**

```
spread_EL (% a.a.) = [(PEC − REC) / VOP] / duration_safra
```

A duration é o saldo devedor médio da safra em anos. Ela converte a base de
desembolso para exposição média e anualiza de uma vez.

---

## 01 — Carteira calibrada

O caso de controle: PD e LGD verdadeiras iguais às precificadas.

**Esperado:** realizado **1,264%** · precificado **1,275%** · gap **−1,1 bps**

Os 1,1 bps são ruído de discretização do hazard, não viés — aparecem em todas as
bases e são o piso de precisão do método. Gap dessa ordem significa modelo
calibrado.

---

## 02 — Subprecificação por PD

PD real subindo ~2% ao mês, preço parado. A produção acelera ao mesmo tempo, que
é o cenário clássico de crescer originando pior.

**Esperado:** realizado **1,578%** · precificado **1,275%** · gap **+30,3 bps**

Gap positivo = a carteira entregou mais perda do que o preço cobrou. Na
decomposição, o `PEC/VOP` ajustado sai acima do que a PD precificada implica e a
LGD implícita bate com a precificada — ou seja, **o erro está na PD**.

---

## 03 — LGD otimista

PD certa (1,700%), mas a LGD precificada em 60% contra 78% realizada: o banco
contava com recuperação que não veio.

**Esperado:** realizado **1,315%** · precificado **1,020%** · gap **+29,5 bps**

Aqui o valor está no expander de decomposição:

| | |
|---|---|
| PEC/VOP ajustado | 1,899% — coerente com PD 1,700%, logo a PD está certa |
| LGD implícita | **78,000%** |
| LGD precificada | **60,000%** |

O gap agregado de 02 e 03 é quase idêntico (+30,3 contra +29,5 bps), mas a causa
é oposta. **O indicador composto não distingue os dois casos; a decomposição
distingue.** É a razão de ela existir.

---

## 04 — Multiproduto

Prazos de 12 a 48 meses na mesma base, **mesmo risco em todos**. Mostra por que
o `(PEC−REC)/VOP` não serve para comparar produtos direto:

| prazo | duration | (PEC−REC)/VOP | EL ajustado |
|---|---|---|---|
| 12m | 0,565 a | 0,839% | 1,275% |
| 24m | 1,127 a | 1,614% | 1,264% |
| 36m | 1,723 a | 2,158% | 1,253% |
| 48m | 2,351 a | 2,440% | 1,243% |

O indicador bruto varia **2,91×** entre o prazo mais curto e o mais longo. O EL
ajustado varia **1,03×**. Toda a diferença do bruto era prazo, não risco.

Quem compara loss rate de consignado 48m com crédito pessoal 12m sem dividir pela
duration conclui que o produto longo é três vezes mais arriscado — e precifica
errado nos dois.

---

## 05 — Safras jovens

Safras de 2025–2026, a maioria sem maturidade. **17 de 22 saem marcadas como
imaturas** e ficam fora do agregado e do gráfico.

| idade | status | bruto | ajustado |
|---|---|---|---|
| 6m | Imatura — não usar | 0,000% | — |
| 10m | Imatura — não usar | 0,000% | — |
| 14m | Imatura — não usar | 0,358% | 1,264% |
| 18m | Em maturação | 0,763% | 1,264% |
| 22m | Em maturação | 1,084% | 1,264% |

Repare nos **0,000%** aos 6 e 10 meses: com lag de 12 meses até a baixa, nenhum
write-off foi lançado ainda. Quem precifica pelo indicador bruto de uma safra
nova cobra **zero** de perda esperada.

Abaixo de 18 meses o módulo recusa o ajuste em vez de extrapolar: o fator de
completude fica pequeno demais e amplifica ruído no lugar de corrigir.

---

## Antes de usar isto para decidir preço

As premissas de maturação (`lag_wo`, `meia_vida_rec`) são **chute plausível**,
não dado. São elas que governam todo o ajuste, e um erro nelas sai direto no
número sem aviso.

Calibre com safras antigas reais: o lag é a distância média entre o default e a
baixa na sua política contábil, e a meia-vida sai da curva de recuperação
observada do seu histórico de write-off.
