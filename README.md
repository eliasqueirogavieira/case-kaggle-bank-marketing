# Quem ligar primeiro? Propensão a depósito a prazo

Case de modelagem sobre o dataset **UCI Bank Marketing** (campanhas de telemarketing de um banco português,
mai/2008–nov/2010). O objetivo é ordenar os clientes de uma campanha pela chance de aderir a um depósito a prazo,
usando só informação disponível **antes** da ligação, e transformar isso em decisões: quem ligar, quantos ligar e
quantas vezes insistir.

## Resumo

- **O "95% de acurácia" é ilusório.** `duration` só existe depois da ligação. Com ela a AUC vai a 0,94; sem ela, e
  sem vazamento no split, o problema é outro. Acurácia também não serve: "nunca ligar" acerta 88,3%.
- **A época domina a conversão.** Ela sai de 5% (2008) para 17% (2009) e 52% (2010). Vários efeitos de "perfil"
  eram época (paradoxo de Simpson): clientes 60+ convertem 2,9× a média na base toda, mas só 1,07× comparados
  com quem foi ligado no mesmo mês.
- **Modelo:** LightGBM que aprende **dentro do mês**, com a taxa do mês como offset no treino. No teste:
  - AUC dentro do mês **0,606**, contra 0,585 da logística (+0,021, IC95% [0,006; 0,036]) e 0,550 de uma regra
    de negócio;
  - na base toda: ROC-AUC 0,805, Gini 0,61, KS 0,49 e PR-AUC 0,465 (ao acaso: 0,117). Essas métricas usam a
    taxa do mês, que antes da campanha precisa ser estimada;
  - no top 20% de cada mês: 29,6% das adesões, contra 27,8% da regra simples. É um ganho modesto no topo.
- **Valor para o negócio:**
  - o top 20% de cada mês captura **~30% das adesões**, e o 1º decil converte 19,3% contra 11,7% na média;
  - a regra `p ≥ C/V`, com a probabilidade ajustada à taxa esperada da campanha, decide **quantos** ligar;
  - da 5ª ligação em diante, o retorno cai para ~0,6× o de uma ligação média do mesmo mês. Com **no máximo 4
    tentativas**, o banco corta 21,8% das ligações e perde 7,4% das adesões.
- **Fora do tempo:** o modelo continua ordenando (AUC no mês 0,63–0,70) e empata com a logística. Sem ajustar a
  probabilidade pela taxa da campanha, perde-se ~69% do resultado.

Todos os números vêm de [`reports/metrics.json`](reports/metrics.json), gerado pelos notebooks e pelo
`bank_marketing.train`.

## Estrutura

```
├── data/raw/                  dados originais (ver data/README.md)
├── notebooks/
│   ├── 01_eda.ipynb                     análise exploratória, ausentes, época, tentativas, vazamento
│   ├── 02_modelagem.ipynb               pré-processamento, métricas, CV, Optuna, ponto de operação, teste, ablação
│   └── 03_interpretacao_producao.ipynb  SHAP, odds ratios, insights, testes fora do tempo, PSI, produção
├── src/bank_marketing/        código reutilizável (usado pelos notebooks e pelos CLIs)
│   ├── config.py      caminhos, seed e contrato de features (quais entram e quais são proibidas)
│   ├── data.py        leitura, reconstrução do período, limpeza, splits aleatório e fora do tempo
│   ├── eda.py         lift na base toda vs. no mesmo mês, taxa por tentativa, impacto do limite de tentativas
│   ├── features.py    pré-processamento (logística e árvores)
│   ├── models.py      LightGBM com offset do mês, logística com efeito fixo, baselines, Optuna
│   ├── evaluation.py  AUC dentro do mês, lift, KS, Gini, bootstrap pareado, PSI, métricas por modelo
│   ├── explain.py     SHAP exato por variável e "motivos" por cliente
│   ├── plots.py       estilo único dos gráficos
│   ├── train.py       CLI: treina o modelo final e registra as métricas
│   └── predict.py     CLI: gera a lista priorizada de ligações
├── models/                    modelo final, metadados e hiperparâmetros
├── reports/                   figuras, métricas e histórico do Optuna
└── tests/                     testes (pytest)
```

## Como reproduzir

Requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.12. No macOS, o LightGBM precisa do OpenMP:
`brew install libomp`.

```bash
git clone https://github.com/eliasqueirogavieira/case-kaggle-bank-marketing.git
cd case-kaggle-bank-marketing

uv sync                      # cria o ambiente com as versões travadas (uv.lock)
uv run pytest                # testes

uv run jupyter lab           # notebooks (os outputs já estão salvos)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # reexecuta tudo (~2 min)

uv run python -m bank_marketing.train              # retreina o modelo final (usa models/best_params.json)
uv run python -m bank_marketing.train --tune 50    # refaz a busca do Optuna (~4 min)

uv run python -m bank_marketing.predict \
    --input data/raw/bank-full.csv --output lista.csv --top 0.2 --base-rate 0.12
```

`predict` recebe clientes no formato do `bank-full.csv` e gera, por cliente: score, posição, decil, flag de
prioridade, a probabilidade (quando a taxa esperada da campanha é informada) e os 3 principais motivos. Colunas
que o modelo não usa, como `duration`, são ignoradas. Sem `uv`, `pip install -r requirements.txt` instala as
mesmas versões.

Tudo usa seed fixa (42). O LightGBM roda em modo determinístico, e CV e Optuna rodam com `n_jobs=1`.

## Decisões principais

| Tema | Decisão | Onde |
|---|---|---|
| Arquivo | `bank-full.csv` original; o `bank_cleaned.csv` remove 4.370 linhas e a coluna `contact` sem documentar | 01 §3 |
| Ausentes | `unknown` mantido como categoria (ausência informativa); `poutcome` → `nonexistent` quando nunca contatado | 01 §2 |
| Outliers | nenhuma remoção (valores reais; o modelo precisa pontuar todos) | 01 §6 |
| Features | só o que se conhece antes da ligação; fora `duration` (vazamento), `campaign` (depende do desfecho), `day`/`month` (época) | 01 §11 |
| Época | offset do mês no LightGBM e efeito fixo na logística: o modelo compara clientes do mesmo mês e não usa canal ou idade como atalho para a época | 02 §2, 03 §1 |
| Métrica | AUC dentro do mês e lift@20% dentro do mês; ROC-AUC, Gini, KS, PR-AUC e Brier como apoio | 02 §3 |
| Desbalanceamento | sem SMOTE nem `class_weight`: a ordenação não precisa, e as probabilidades ficam calibradas | 02 §3 |
| Validação | hold-out 80/20 estratificado + CV 5 folds + 2 testes fora do tempo | 02, 03 §5 |
| Decisão de negócio | ordenar pelo score; ligar enquanto `p ≥ C/V`, com a probabilidade calibrada para a taxa esperada da campanha; máximo de 4 tentativas | 01 §9, 02 §6, 03 §6 |

## Limitações e próximos passos

- **Sem ID de cliente:** as linhas são cliente × campanha, então a mesma pessoa pode estar no treino e no teste
  do split aleatório. Os testes fora do tempo reduzem esse risco.
- **Viés de seleção:** só observamos quem o banco escolheu ligar. Em produção, um grupo de controle aleatório
  corrige isso e mede o ganho incremental.
- **Premissas:** C = €10 e V = €100 são ilustrativas; com custos e margens reais o limiar muda.
- **Próximos passos:** variáveis macro do `bank-additional` (Euribor, emprego), modelo de *uplift* e A/B
  champion/challenger (LightGBM vs. logística vs. regra).

## Dados

Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.*
Decision Support Systems, 62, 22–31. Dataset: UCI Machine Learning Repository (CC BY 4.0), via Kaggle.
