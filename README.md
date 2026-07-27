# Interest Rate Loan Pricing — RAROC (Streamlit)

Calculadora de precificação de crédito ajustada ao risco (modelo RAROC, consignado INSS).
Porte do app Shiny para **Streamlit**, para rodar de graça no Streamlit Community Cloud.

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar de graça (Streamlit Community Cloud)
1. Suba `app.py` e `requirements.txt` para a raiz de um repositório no GitHub.
2. Acesse https://share.streamlit.io e conecte a conta do GitHub.
3. "Create app" → Repository, Branch `main`, Main file path `app.py` → **Deploy**.
4. Em ~2 min o app fica no ar num link `*.streamlit.app` (gratuito, HTTPS).

## Metodologia
O app tem um seletor **"Metodologia de cálculo"**:

- **Consistente (padrão):** incorpora as correções de validação —
  (1) spread UL apenas sobre o prêmio de capital; (2) base temporal anual = anual
  (sem o fator n/12 no alvo); (3) comissão/custos integrais (one-time).
  No caso base out/24 → **2,16% a.m.**
- **Legado:** reproduz exatamente os números do app Shiny / dissertação
  (spread EL corrigido, demais itens como no original). Caso base → **2,33% a.m.**

O resultado mostra o número da metodologia ativa e, como referência, o que a outra daria.

## Sem banco de dados / login
A versão Shiny usava Supabase para paywall (com senha no código — removida aqui).
Se precisar de acesso restrito, use `st.secrets` + `streamlit-authenticator`;
nunca coloque credenciais no código-fonte.
