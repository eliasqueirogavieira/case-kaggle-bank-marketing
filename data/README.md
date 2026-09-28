# Dados

| Arquivo | Origem | Uso |
|---|---|---|
| `raw/bank-full.csv` | UCI Bank Marketing (`bank-full.csv`), via [Kaggle](https://www.kaggle.com/datasets/abdelazizsami/bank-marketing/data) | **fonte única da análise** |
| `raw/bank_cleaned.csv` | versão derivada publicada no Kaggle | não usada; mantida para mostrar por quê (notebook 01, §3) |

- `bank-full.csv`: 45.211 linhas × 17 colunas, separador `;`, sem valores nulos (a ausência aparece como `unknown`).
- Segundo a documentação da UCI, as linhas estão **ordenadas por data**, de maio/2008 a novembro/2010. O ano é
  reconstruído a partir dessa ordem em `bank_marketing.data.add_period`.
- Licença: CC BY 4.0. Citação: Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the
  success of bank telemarketing.* Decision Support Systems, 62, 22–31.

## Dicionário

| Variável | Descrição | Uso no modelo |
|---|---|---|
| `age` | idade | sim |
| `job` | tipo de trabalho | sim |
| `marital` | estado civil (`divorced` inclui viúvos) | sim |
| `education` | escolaridade | sim |
| `default` | tem crédito em inadimplência? | sim |
| `balance` | saldo médio anual (€) | sim |
| `housing` | tem crédito imobiliário? | sim |
| `loan` | tem empréstimo pessoal? | sim |
| `contact` | canal do contato (celular, fixo, não registrado) | sim |
| `day`, `month` | data do último contato | não (época; ver notebook 01) |
| `duration` | duração da última ligação (s) | não (só existe depois da ligação) |
| `campaign` | nº de contatos nesta campanha, incluindo o último | não (depende do desfecho; vira a regra de tentativas) |
| `pdays` | dias desde o contato numa campanha anterior (−1 = nunca) | sim (+ flag `contacted_before`) |
| `previous` | nº de contatos antes desta campanha | sim |
| `poutcome` | resultado da campanha anterior | sim (`unknown` → `nonexistent` quando nunca contatado) |
| `y` | aderiu ao depósito a prazo? | alvo |
