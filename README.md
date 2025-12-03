# Python_Modelos_Predicao_Estatistica
Análise comparativa de algoritmos essenciais de Machine Learning para Regressão e Classificação usando Python (Scikit-learn, Pandas) e Jupyter Notebooks.

### ⚖️ Análise Comparativa dos Modelos de Regressão Base (Python/Scikit-learn)

#### Regressão Linear Múltipla (`mpg ~ wt + hp`)

O treinamento em Python (utilizando `statsmodels` para análise estatística detalhada) replicou com sucesso os resultados obtidos em R, confirmando a validade do modelo:

| Métrica | Resultado |
| :--- | :--- |
| **R-quadrado Ajustado** | $\mathbf{0.8148}$ |
| **RSE (RMSE)** | $\mathbf{2.593}$ |
| **Coeficiente do Peso (`wt`)** | $-3.8778$ |
| **P-valor do `hp`** | $0.001$ |

**Previsão de Exemplo:**

| Parâmetros de Entrada | Previsão de MPG |
| :--- | :--- |
| $\text{Peso} = 3.000$ lbs, $\text{HP} = 150$ | $\mathbf{20.83}$ |

**Conclusão:** O modelo é estatisticamente robusto, com $81.48\%$ da variância de MPG explicada, e o efeito da multicolinearidade entre Peso e Horsepower é corretamente contabilizado.
