# Previsão de séries temporais: Random Forest vs. MLP

**Reconstrução em 2026 de um exercício acadêmico discutido em maio de 2023.** Na época, o enunciado solicitava selecionar um conjunto de dados de série temporal, usar Random Forest para identificar variáveis importantes, treinar uma rede neural e comparar as previsões pelo erro MAPE. O histórico recuperado contém o enunciado e a discussão, **não uma execução original comprovada**.

Este repositório foi preparado do zero como demonstração reproduzível. Ele **não corresponde a código original de 2023** e não apresenta resultados sintéticos como desempenho em produção.

## Experimento

- Gera uma série diária **sintética** com tendência, sazonalidade semanal e ruído.
- Cria atributos com dados passados: valor do dia anterior, valor de sete dias antes, média móvel de sete dias anteriores e dia da semana.
- Separa os primeiros 80% dos registros para treino e os últimos 20% para teste, preservando a sequência temporal.
- Treina Random Forest, calcula importância de atributos **somente no treino** e seleciona os três primeiros para a MLP.
- Compara Random Forest e MLP usando **MAPE e MAE**; exporta previsões e importância dos atributos.

## Executar

Python 3.10+ recomendado.

```bash
python -m pip install -r requirements.txt
python -m src.forecast
```

Arquivos gerados localmente: `results/metrics.csv`, `results/predictions.csv` e `results/feature_importance.csv`.

Para testar:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## O que aprendi / limitações

- Em séries temporais, dividir aleatoriamente treino/teste pode criar avaliações excessivamente otimistas; neste exemplo a divisão respeita o tempo.
- Os atributos defasados não incorporam o valor real do dia a prever, evitando um vazamento imediato do alvo.
- As importâncias do Random Forest **não demonstram causalidade** e podem ser instáveis com atributos correlacionados.
- Esta é uma avaliação de previsão **um passo à frente**, que usa observações anteriores disponíveis. Não é previsão recursiva de vários dias sem valores observados.
- O enunciado de 2023 também citava **LSTM** em uma comparação final, mas este primeiro exemplo implementa apenas Random Forest e MLP. Não há resultados de LSTM alegados ou inventados.
- Os dados são fictícios; o desempenho não indica qualidade em um cenário real.

## Origem

- **2023:** exercício de regressão temporal, importância de variáveis, MAPE, Random Forest e redes neurais.
- **2026:** implementação reproduzível, cuidados de validação temporal, documentação e testes.
