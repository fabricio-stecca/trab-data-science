Divisão `1ff36d32de` — melhor configuração de cada algoritmo escolhida pela validação (`cost_mean`); métricas abaixo são do **teste**.

| algoritmo | configuração escolhida | nº configs | val_cost_mean | limiar | acurácia | acurácia_bal. | precisão | recall | especificidade | f1 | auc | custo_médio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | sweep__activation=relu__optimizer=adamw__hidden_layers=32-16 | 9 | 0.495 | 0.290 | 0.480 | 0.600 | 0.355 | 0.900 | 0.300 | 0.509 | 0.770 | 0.640 |
| KNN | knn__n_neighbors=15__weights=distance | 6 | 0.585 | 0.140 | 0.490 | 0.598 | 0.356 | 0.867 | 0.329 | 0.505 | 0.713 | 0.670 |
| Random Forest | random_forest__max_depth=None__min_samples_leaf=5 | 4 | 0.475 | 0.340 | 0.610 | 0.688 | 0.427 | 0.883 | 0.493 | 0.576 | 0.777 | 0.530 |
| Gradient Boosting | gradient_boosting__learning_rate=0.03__max_depth=3 | 4 | 0.515 | 0.260 | 0.565 | 0.661 | 0.400 | 0.900 | 0.421 | 0.554 | 0.768 | 0.555 |
