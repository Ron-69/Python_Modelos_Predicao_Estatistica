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

### 📐 Análise de Forma Funcional: Linear vs. Polinomial

Esta seção compara o modelo Linear Múltiplo (que foi o melhor ajuste linear) com os modelos Polinomiais de 2º e 3º Graus (`mpg` ~ $\text{hp} + \text{hp}^2$, etc.), a fim de determinar a melhor forma de modelar a relação.

#### Comparativo Consolidado de Desempenho (mtcars)

| Modelo | Fórmula | R-quadrado Ajustado | RSE (RMSE) | Termos Polinomiais Significativos ($\text{P} < 0.05$) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Múltiplo** | $\text{mpg} \sim \text{wt} + \text{hp}$ | $\mathbf{0.8148}$ | $\mathbf{2.593}$ | N/A |
| **Polinomial 2º Grau** | $\text{mpg} \sim \text{hp} + \text{hp}^2$ | $0.7393$ | $3.077$ | $\text{hp}^2$ (**Sim**) |
| **Polinomial 3º Grau** | $\text{mpg} \sim \text{hp} + \text{hp}^2 + \text{hp}^3$ | $0.7349$ | $3.103$ | Nenhum ($\text{hp}^2$ e $\text{hp}^3$ não significativos) |

#### Conclusão sobre a Modelagem

1.  **Significância da Curvatura:** O termo quadrático ($\text{hp}^2$) foi estatisticamente significativo no modelo de 2º grau, provando que a relação entre $\text{HP}$ e $\text{MPG}$ **não é estritamente linear**.
2.  **Modelo Preditivo Vencedor:** Apesar da comprovação da curvatura, o modelo **Linear Múltiplo** ($\text{mpg} \sim \text{wt} + \text{hp}$) se mostrou o **melhor modelo preditivo** em termos de ajuste ($\mathbf{R^2_{adj}=0.8148}$) e precisão ($\mathbf{RSE=2.593}$).
3.  **Estratégia Ideal:** Para o *dataset* `mtcars`, a **combinação de *features* independentes** ($\text{wt}$ e $\text{hp}$) foi significativamente mais eficaz para reduzir o erro de previsão do que tentar modelar a forma não-linear de um único *feature* ($\text{hp}$). O modelo Polinomial de 3º Grau, além de não ser significativo, teve o pior desempenho.

---
