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
### 📈 Regularização de Modelos (Ridge, Lasso e Elastic Net)

Os modelos de regularização foram aplicados ao *dataset* California Housing para otimizar a performance, prevenir o *overfitting* e realizar a seleção de *features*.

#### Comparativo de Desempenho (California Housing)

| Modelo | Penalidade | Melhor Alpha ($\lambda$) | L1 Ratio ($\alpha$) | RMSE (Teste) | R² (Teste) | Seleção de Features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge (L2)** | $L2$ | $4.3288$ | $0.0000$ | $\mathbf{0.5305}$ | $0.5959$ | Encolhe (não zera) |
| **Lasso (L1)** | $L1$ | $0.0027$ | $1.0000$ | $0.7270$ | $0.5973$ | $\text{Population}$ zerada |
| **Elastic Net** | $L1 + L2$ | $0.0027$ | $\mathbf{1.0000}$ | $0.7270$ | $0.5973$ | $\text{Population}$ zerada |

#### Conclusão Final da Regularização

1.  **Modelo Vencedor:** A **Regressão Ridge (L2)** se mostrou o **modelo preditivo superior**, com o menor erro (RMSE de $\mathbf{0.5305}$). Isso sugere que, para este *dataset*, é melhor manter todas as *features*, apenas encolhendo seus pesos.
2.  **Lasso e Elastic Net:** Ambos otimizaram para o mesmo ponto (Elastic Net otimizou para ser Lasso puro, $\mathbf{L1\_Ratio=1.0}$), zerando o peso da variável $\text{Population}$. No entanto, essa remoção resultou em uma perda significativa na precisão ($\text{RMSE}$ $\mathbf{\approx 37\%}$ maior).

O modelo **Ridge** será o modelo escolhido para a fase de *deployment* e produção, devido à sua precisão superior.

## 📊 Diferença de Resultados Devido à Troca de Datasets

Os resultados da regularização (Ridge, Lasso e Elastic Net) obtidos em **Python (California Housing)** e em **R (Boston Housing)** apresentaram uma diferença significativa na escolha do modelo de penalidade ideal.

Essa divergência é causada pela **diferença fundamental** entre os dois *datasets* utilizados: o **tamanho amostral** e o **contexto dos dados**.

---

### 1. Comportamento Ideal da Penalidade por Dataset

| Característica | Boston Housing (R) | California Housing (Python) |
| :--- | :--- | :--- |
| **Tamanho Amostral ($N$)** | Pequeno ($N=506$) | Grande ($\mathbf{N=20.640}$) |
| **Ideal de Regularização** | **Elastic Net (L2-Dominante)** | **Ridge (L2)** |
| **Melhor L1 Ratio ($\alpha$)** | $\mathbf{0.1111}$ ($\approx 90\%$ L2) | $\mathbf{0.0000}$ (Ridge Puro) |
| **Motivo** | Prioriza a **estabilidade (L2)** em *datasets* pequenos, pois a exclusão de *features* pelo Lasso é arriscada. | Prioriza a **estabilidade (L2)** para o menor erro. O Lasso teve perda preditiva significativa ao remover o *feature* `Population`. |
| **Vencedor Final** | Elastic Net (RMSE: 5.179) | **Ridge** (RMSE: 0.5305) |

---

### 2. Implicações da Divergência

O Elastic Net é projetado para encontrar a melhor **mistura** ($\alpha$) de penalidades L1 e L2.

* No **Boston Housing (R)**, o Elastic Net otimizou para um modelo **majoritariamente Ridge** ($\alpha \approx 0.11$), confirmando que a penalidade L2 (estabilidade) é mais importante.
No **California Housing (Python)**, o Lasso e o Elastic Net (que otimizou para ser Lasso Puro: $\text{L1}_{\text{Ratio}}=1.0$) tiveram um desempenho **muito inferior** ao Ridge (L2).

**Conclusão Consolidada:**

Em ambos os *datasets*, a abordagem vencedora foi **priorizar a penalidade Ridge (L2)**, que **encolhe** os coeficientes sem zerá-los.

* O **Boston Housing** exigiu o Elastic Net para encontrar essa **dominância L2**.
* O **California Housing** demonstrou que o **Ridge Puro** é o mais robusto e preditivo, confirmando que a estabilidade é a chave para o melhor desempenho em ambos os contextos.

---

### 🚀 Modelos Não Linear e Métodos de Ensemble

Os modelos de Regressão Não Linear (KNN) e os Métodos de Ensemble (Random Forest e XGBoost) foram aplicados ao *dataset* California Housing para capturar relações mais complexas e atingir maior precisão. Já os modelos de Classificação Não Linear (KNN) e os modelos de Ensemble (Random Forest e XGBoost) foram aplicado ao *dataset* Pima Indian Diabetes.

## ⚙️ Modelos Não Lineares e Ensembles: Versatilidade na Modelagem

Os modelos de Machine Learning (ML) usados neste projeto, nomeadamente **K-Nearest Neighbors (KNN)** e os métodos de **Ensemble** (**Random Forest** e **XGBoost**), são notáveis por sua **versatilidade**.

Eles são chamados de "modelos de propósito geral" porque suas estruturas podem ser adaptadas para resolver problemas de **Regressão** (previsão de um valor contínuo) e **Classificação** (previsão de uma categoria discreta) sem a necessidade de assumir relações lineares.

### Justificativa da Estrutura de Notebooks

A distinção entre as tarefas é feita na **função de agregação** final do algoritmo:

| Tipo de Problema | O que o Modelo Previsível? | Função Final do Algoritmo | Notebook Correspondente |
| :--- | :--- | :--- | :--- |
| **Regressão** | Um **Valor Contínuo** (Ex: Preço de Casa) | **Média** das previsões das árvores ou vizinhos. | `02_Regression_NonLinear_and_Ensembles.ipynb` |
| **Classificação** | Uma **Categoria Discreta** (Ex: Diabetes Sim/Não) | **Voto Majoritário** ou **Média das Probabilidades** (usando um limite de corte). | `02_Classification_NonLinear_and_Ensembles.ipynb` |

Esta separação em *notebooks* dedicados garante que as métricas de avaliação e as técnicas de otimização (focadas em **RMSE/R²** para Regressão e **AUC-ROC/Acurácia** para Classificação) sejam tratadas de forma independente e adequada.

### 🚀 Regressão Não Linear(KNN) e Métodos de Ensemble (Random Forest e XGBoost)

#### Comparativo de Desempenho (California Housing)

A tabela abaixo resume os resultados de desempenho em comparação com o modelo Linear mais forte (Ridge):

| Modelo | Tipo | Melhor Parâmetro | R² (Teste) | RMSE (Teste) | Variação R² (vs. Ridge) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge (L2)** | Linear | $\lambda=4.3288$ | $0.5959$ | $0.5305$ | Base Linear |
| **KNN** | Não Linear | $K=11$ | $0.6869$ | $0.6411$ | $+9.1$ p.p. |
| **Random Forest** | Ensemble | $n_{est}=200, depth=20$ | $0.8060$ | $0.5046$ | $+21.0$ p.p. |
| **XGBoost** | **Ensemble (Boosting)** | $lr=0.1, n_{est}=200, depth=5$ | $\mathbf{0.8358}$ | $\mathbf{0.4642}$ | $\mathbf{+24.0}$ p.p. |

#### Conclusão Global: Modelo Preditivo Vencedor

O **XGBoost (Gradient Boosting)** demonstrou ser o modelo mais eficaz:

1.  **Melhor Explicação (R²):** Explica $\mathbf{83,58\%}$ da variância no preço das casas, indicando uma excelente capacidade de modelar as relações complexas do *dataset*.
2.  **Melhor Precisão (RMSE):** Obteve o menor erro médio de previsão ($\mathbf{0.4642}$), superando todos os outros modelos testados, incluindo o Random Forest.

A estratégia de **Gradient Boosting** será a base para as previsões finais do projeto.

---
### 🌳 Classificação Não Linear(KNN) e Métodos de Ensemble (Random Forest e XGBoost)

O **XGBoost (Gradient Boosting)** e o **Random Forest** foram aplicados ao *dataset* Pima Indians Diabetes para explorar o poder dos métodos Não_Linear e de árvore no problema de classificação.

### 🌐 Classificação com K-Nearest Neighbors (KNN)

O modelo KNN, classificado como Não Linear, foi treinado para determinar seu poder preditivo no *dataset* Pima Indians Diabetes.

#### Resultados do KNN Classifier

| Métrica | Resultado |
| :--- | :--- |
| **Melhor Parâmetro** | $K=21$ (weights: 'distance') |
| **Acurácia (Teste)** | $\mathbf{0.7706}$ |
| **AUC-ROC (Teste)** | $0.8127$ |
| **Recall (Classe 1 - Diabetes)** | $0.54$ |

#### Conclusão do KNN

O KNN alcançou a **maior Acurácia ($\mathbf{77.06\%}$) de todos os modelos** testados. O modelo se beneficia de um grande número de vizinhos ($K=21$), sugerindo que a fronteira de decisão é relativamente suave e que a votação por distância (peso maior para vizinhos mais próximos) é a mais eficaz.

#### Comparativo de Desempenho (Ensemble)

| Modelo | Parâmetros Otimizados | AUC-ROC (Teste) | Acurácia (Teste) | Recall (Classe 1 - Diabetes) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | $n_{est}=100, depth=5$ | $0.8305$ | $0.7359$ | $0.49$ |
| **XGBoost** | $lr=0.05, n_{est}=100, depth=3$ | $\mathbf{0.8416}$ | $\mathbf{0.7576}$ | $0.54$ |

#### Conclusão do Ensemble

O **XGBoost** demonstrou ser o modelo de árvore mais poderoso para este problema, superando o Random Forest em todas as métricas gerais de desempenho.

---

### 💉 Modelos Probabilísticos para Classificação (Pima Indians Diabetes)

O *dataset* Pima Indians Diabetes foi utilizado para a classificação binária (Diabetes: Sim/Não), aplicando modelos que estimam probabilidades.

#### Comparativo de Desempenho (Pima Indians Diabetes - Probabilísticos)

| Modelo | Penalidade | Métrica de Otimização | AUC-ROC (Teste) | Acurácia (Teste) | Recall (Classe 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Nenhuma | N/A | $0.8088$ | $0.7446$ | $\mathbf{0.62}$ |
| **Regressão Logística** | L2 ($C=1.0$) | AUC-ROC | $\mathbf{0.8380}$ | $\mathbf{0.7446}$ | $0.52$ |

#### Conclusão Parcial

1.  **Regressão Logística** demonstrou ser superior em **capacidade de distinção** entre as classes (maior **AUC-ROC: 0.8380**).
2.  O **Naive Bayes** apresentou um **Recall** superior para a classe alvo (Diabetes: $\mathbf{0.62}$), indicando que ele é mais eficaz em capturar casos positivos reais (menos falsos negativos).

A **Regressão Logística** é o modelo probabilístico com melhor desempenho geral.

---
### 💉 Modelos de Classificação (Pima Indians Diabetes)

O objetivo desta etapa foi classificar se um paciente indígena Pima seria diagnosticado com diabetes (Classe 1), utilizando modelos Probabilísticos, Não Lineares (KNN) e de Ensemble.

#### Resultados Consolidados

| Modelo | Categoria | Melhor Parâmetro | AUC-ROC (Teste) | Acurácia (Teste) | Recall (Classe 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Probabilístico | N/A | $0.8088$ | $0.7446$ | $\mathbf{0.62}$ |
| **Regressão Logística** | Probabilístico | $C=1.0$ | $0.8380$ | $0.7446$ | $0.52$ |
| **KNN** | **Não Linear** | $K=21, weights=distance$ | $0.8127$ | $\mathbf{0.7706}$ | $0.54$ |
| **Random Forest** | Ensemble | $n_{est}=100, depth=5$ | $0.8305$ | $0.7359$ | $0.49$ |
| **XGBoost** | **Ensemble (Boosting)** | $lr=0.05, n_{est}=100, depth=3$ | $\mathbf{0.8416}$ | $0.7576$ | $0.54$ |

#### Conclusão Geral da Classificação

Os modelos de classificação apresentam um *trade-off* claro:

1.  **Melhor Capacidade de Distinção (AUC-ROC):** O **XGBoost** é o vencedor ($\mathbf{0.8416}$), sendo o mais eficaz em ranquear corretamente as probabilidades de diabetes.
2.  **Melhor Precisão Geral (Acurácia):** O **KNN** alcança a maior acurácia ($\mathbf{0.7706}$), sendo o modelo que mais frequentemente acerta a previsão final.
3.  **Melhor Identificação de Positivos (Recall):** O **Naive Bayes** é o mais adequado se a prioridade for **minimizar Falsos Negativos** (Recall: $\mathbf{0.62}$).

O **XGBoost** é o modelo de melhor performance geral (AUC-ROC), mas o **KNN** oferece a maior taxa de acerto.

---
### 📏 Modelo de Margem: Support Vector Machine (SVM)

O **SVM** (SVC) busca o hiperplano que maximiza a margem entre as classes. A otimização selecionou um kernel linear com alta penalidade de erro, similar aos modelos lineares tradicionais.

#### Resultados do SVM Classifier

| Métrica | Resultado |
| :--- | :--- |
| **Melhor Parâmetro** | $C=10.0$, **kernel**: **`linear`** |
| **Acurácia (Teste)** | $0.7359$ |
| **AUC-ROC (Teste)** | $0.8346$ |
| **Recall (Classe 1 - Diabetes)** | $0.49$ |

#### Conclusão do SVM

O SVM, com seu kernel linear, demonstrou um desempenho de distinção (**AUC-ROC: 0.8346**) muito próximo ao da Regressão Logística, mas com um baixo Recall, indicando que a separação linear é eficaz, porém limitada na identificação dos casos positivos mais difíceis.

---

### 🌲 Árvore de Decisão (Decision Tree Classifier)

A Árvore de Decisão, como modelo fundamental para os métodos de Ensemble, foi otimizada para identificar a profundidade ideal no problema de classificação.

#### Resultados da Árvore de Decisão

| Métrica | Resultado |
| :--- | :--- |
| **Melhor Parâmetro** | $max\_depth=5, min\_samples\_split=10$ |
| **Acurácia (Teste)** | $0.7619$ |
| **AUC-ROC (Teste)** | $0.8149$ |
| **Recall (Classe 1 - Diabetes)** | $0.44$ |

#### Conclusão da Árvore

Embora a acurácia seja alta ($\approx 76.2\%$), o baixo Recall da Classe 1 ($0.44$) indica que o modelo de árvore pura tem dificuldade em generalizar os casos positivos de forma isolada, justificando a necessidade dos métodos de Ensemble (Random Forest e XGBoost) para melhorar a estabilidade e a performance.

---

### 💉 Modelos de Classificação (Pima Indians Diabetes)

A análise comparativa final no *dataset* Pima Indians Diabetes incluiu 7 modelos de diferentes categorias para a classificação binária de diabetes.

#### Resultados Consolidados Finais

| Modelo | Categoria | Melhor Parâmetro | AUC-ROC (Teste) | Acurácia (Teste) | Recall (Classe 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Probabilístico | N/A | $0.8088$ | $0.7446$ | $\mathbf{0.62}$ |
| **Regressão Logística** | Probabilístico | $C=1.0$ | $0.8380$ | $0.7446$ | $0.52$ |
| **Decision Tree** | **Árvore** | $max\_depth=5, min\_samples\_split=10$ | $0.8149$ | $0.7619$ | $0.44$ |
| **KNN** | Não Linear | $K=21, weights=distance$ | $0.8127$ | $\mathbf{0.7706}$ | $0.54$ |
| **SVM (SVC)** | Margem/Distância | $C=10.0, kernel=linear$ | $0.8346$ | $0.7359$ | $0.49$ |
| **Random Forest** | Ensemble | $n_{est}=100, depth=5$ | $0.8305$ | $0.7359$ | $0.49$ |
| **XGBoost** | Ensemble (Boosting) | $lr=0.05, n_{est}=100, depth=3$ | $\mathbf{0.8416}$ | $0.7576$ | $0.54$ |

#### Conclusão Geral da Classificação (Final)

O modelo de melhor desempenho é o **XGBoost**, que alcançou o maior **AUC-ROC ($\mathbf{0.8416}$)**, indicando a melhor capacidade de ranqueamento e distinção entre as classes.

* **Para Distinção e Performance Geral (AUC-ROC):** O **XGBoost** é o vencedor.
* **Para Melhor Precisão Geral (Acurácia):** O **KNN** possui a maior taxa de acerto ($\mathbf{0.7706}$).
* **Para Minimizar Falsos Negativos (Recall):** O **Naive Bayes** é o modelo mais sensível aos casos positivos de diabetes ($\mathbf{0.62}$).
---
### 🏆 Stacking Ensemble (Otimização Final do AUC-ROC)

A etapa final do projeto de classificação foi a implementação do **Stacking Ensemble** (Generalização Empilhada) para tentar superar o melhor AUC-ROC individual obtido pelo XGBoost. Esta técnica profissional combina as previsões de modelos heterogêneos, utilizando um **Meta-Modelo** (Regressão Logística) para aprender a ponderar as forças e fraquezas de cada modelo base.

#### Modelos Base (Base Learners)

O Ensemble foi construído utilizando os quatro modelos mais performáticos de diferentes famílias:

* **XGBoost:** Melhor ranqueamento (AUC-ROC).
* **Regressão Logística:** Bom ranqueamento e modelo linear.
* **KNN:** Melhor acurácia (Modelo não linear baseado em distância).
* **Naive Bayes:** Melhor recall (Modelo probabilístico).

#### Metodologia Stacking

1.  Os modelos de base foram carregados e configurados para produzir **probabilidades** (*predict\_proba*).
2.  O **`StackingClassifier`** foi treinado, utilizando 5-fold cross-validation (`cv=5`) para gerar o novo conjunto de dados de previsões.
3.  O **Meta-Modelo** (`LogisticRegression`) foi ajustado neste novo conjunto de previsões.

#### Resultado Final do Stacking Ensemble

O Stacking Ensemble demonstrou uma melhoria no AUC-ROC, estabelecendo um novo patamar de desempenho para o projeto:

| Modelo | Métrica Otimizada | AUC-ROC (Teste) | Acurácia (Teste) | Recall (Classe 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** (Melhor Individual) | AUC-ROC | $0.8416$ | $0.7576$ | $0.54$ |
| **Stacking Ensemble** | AUC-ROC | $\mathbf{0.8421}$ | $0.74$ | $0.52$ |

**Conclusão Final do Projeto:**

O **Stacking Ensemble** é o **modelo final** recomendado para o problema de classificação binária no dataset Pima Indians Diabetes, pois alcançou o maior poder de distinção entre as classes ($\mathbf{0.8421}$). Embora o ganho tenha sido marginal, ele confirma a robustez do Ensemble.

| Objetivo de Negócio | Modelo Recomendado | Métrica |
| :--- | :--- | :--- |
| **Máxima Triagem/Ranqueamento de Risco** | **Stacking Ensemble** | **AUC-ROC** |
| **Máxima Detecção de Positivos (Sensibilidade)** | **Naive Bayes** | **Recall** ($\mathbf{0.62}$) |