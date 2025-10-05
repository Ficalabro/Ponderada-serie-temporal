# 🌦️ Previsão de Temperatura Diária — São Paulo, Brasil

Este projeto implementa e compara dois modelos de **previsão de séries temporais** utilizando dados climáticos diários da cidade de **São Paulo**, disponíveis publicamente no Kaggle.  
O objetivo é analisar o desempenho de abordagens **estatísticas** (Prophet) e **baseadas em aprendizado profundo** (LSTM) na predição de temperatura média diária.

---

## 📊 Dataset

**Fonte:** [Global Daily Climate Data — Kaggle](https://www.kaggle.com/datasets/guillemservera/global-daily-climate-data)

O dataset contém dados meteorológicos diários de diversas cidades do mundo, incluindo:
- Temperaturas médias, mínimas e máximas;
- Precipitação e neve;
- Direção e velocidade do vento;
- Pressão atmosférica e duração de luz solar.

Neste notebook, foi utilizada apenas a cidade de **São Paulo (Brasil)**, com foco na coluna `avg_temp_c` (temperatura média em °C).

---

## 🧹 Preparação dos Dados

O dataframe original contém múltiplas estações meteorológicas e diversas lacunas temporais.  
Para garantir consistência, foram aplicadas as seguintes etapas de saneamento:

1. **Filtragem por cidade:** seleção apenas de registros com `"Sao Paulo"` ou `"São Paulo"`;
2. **Agregação diária:** cálculo da **média da temperatura** por dia;
3. **Reamostragem temporal:** conversão da série para frequência **diária (D)**;
4. **Interpolação temporal:** preenchimento de valores ausentes via interpolação linear baseada no tempo.

O resultado é uma série temporal contínua de temperatura média diária de **1940 até 2024**, sem valores ausentes.

---

## 🔮 Modelos Utilizados

### 1️⃣ Prophet (Facebook Prophet)
Modelo estatístico aditivo, adequado para séries temporais com **tendência + sazonalidade**.  
Configuração usada:
- `daily_seasonality=True`
- `yearly_seasonality=True`
- Previsão no período de teste (20 % finais dos dados).

### 2️⃣ LSTM (Long Short-Term Memory)
Rede neural recorrente projetada para capturar **dependências de longo prazo** em séries temporais.

**Pipeline:**
- Normalização com `MinMaxScaler`;
- Janelas deslizantes de 15 dias;
- Arquitetura: duas camadas LSTM (64 → 32 neurônios) + Dropout 0.1;
- Otimizador Adam e função de perda MSE;
- Early Stopping com pacience = 5 épocas.

---

## ⚙️ Estrutura do Notebook

| Etapa | Descrição |
|:--|:--|
| **1.** Importação e instalação de dependências | Prophet, TensorFlow, Seaborn etc. |
| **2.** Download do dataset via `kagglehub` | Download automático da versão mais recente. |
| **3.** Limpeza e preparo dos dados | Filtragem, agrupamento, reamostragem e interpolação. |
| **4.** Treinamento Prophet | Criação, treinamento e previsão. |
| **5.** Treinamento LSTM | Criação de janelas, treinamento e previsão. |
| **6.** Visualizações | Gráficos das previsões individuais e comparativas. |
| **7.** Avaliação de métricas | RMSE e MAE comparando Prophet e LSTM. |

---

## 📈 Resultados e Comparação

| Modelo | RMSE (°C) | MAE (°C) |
|:--|:--:|:--:|
| Prophet | ~ X °C | ~ Y °C |
| LSTM | ~ X °C | ~ Y °C |

*(valores variam conforme a execução e o período de treino)*

### ✅ Principais Observações
- O **Prophet** apresenta bom desempenho em capturar **sazonalidade anual**.  
- A **LSTM** é capaz de representar padrões não lineares, mas pode exigir mais dados e ajuste de hiperparâmetros.  
- O uso conjunto das métricas **RMSE** e **MAE** permite avaliar precisão e robustez.  

---

## 📐 Métricas de Avaliação

- **RMSE (Root Mean Squared Error):** penaliza erros maiores, enfatizando previsões distantes do valor real.  
- **MAE (Mean Absolute Error):** fornece a média absoluta dos erros em °C, de interpretação direta.

📚 **Referência:**  
> Hyndman, R. J. & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* — Cap. 5.7.

---

## 💻 Execução

1. Abra o notebook no **Google Colab**;  
2. Execute célula por célula, na ordem;  
3. Aguarde o download via `kagglehub`;  
4. Observe os gráficos e as métricas finais.

---

## 📦 Dependências

```bash
pip install kagglehub prophet tensorflow scikit-learn matplotlib seaborn pandas pyarrow
