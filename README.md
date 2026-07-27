# Interest Rate Loan Pricing — RAROC (Streamlit)

Calculadora de precificação de crédito ajustada ao risco (modelo RAROC, consignado INSS).
Porte do app Shiny para **Streamlit**, para rodar de graça no Streamlit Community Cloud.

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar de graça (Streamlit Community Cloud)
1. Crie um repositório no GitHub com `app.py` e `requirements.txt`.
2. Acesse https://share.streamlit.io e conecte a conta do GitHub.
3. "New app" → escolha o repositório, branch `main`, arquivo `app.py` → **Deploy**.
4. Em ~2 min o app fica no ar num link `*.streamlit.app` (gratuito, HTTPS).

## Observações do modelo
- O **padrão reproduz exatamente** os números do app Shiny atual.
- O expander "Ajustes de consistência" aplica, opcionalmente, as correções dimensionais
  discutidas na validação (spread UL só sobre o prêmio; base temporal anual=anual;
  comissão/custos integrais). Desligados = comportamento idêntico ao Shiny.
- **Sem banco de dados / login**: a versão Shiny usava Supabase para paywall. Aqui o
  cálculo é aberto. Se precisar de acesso restrito, use `st.secrets` + o pacote
  `streamlit-authenticator` (não coloque senha no código).
