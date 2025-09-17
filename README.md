# Modelo de Regressão Linear com Validação Cruzada

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/stable/)

Este repositório contém um trabalho acadêmico que implementa um modelo de regressão linear simples utilizando o dataset `aerogerador.dat`. O objetivo é prever a potência gerada com base na velocidade do vento, utilizando validação cruzada para avaliar o desempenho do modelo.

---

## 🚀 Quick Start

```bash
# Clone o repositório e instale as dependências
pip install numpy pandas matplotlib seaborn scikit-learn

# Execute o script principal
python modelo_de_regressao_linear.py
```

**Output:** Resultados salvos em `teste_de_resultados_do_modelo/resultados_metricas.txt`

---

## 🎯 Features & Funcionalidades

| Feature | Descrição |
|---------|-----------|
| 📈 **Treinamento e Validação** | Modelo de regressão linear com validação cruzada |
| 📊 **Visualização de Métricas** | Histogramas de MSE e MAE |
| 💾 **Resultados Salvos** | Métricas armazenadas em arquivo TXT |
| 🔍 **Configuração Personalizável** | Número de rodadas e divisão treino/teste |

---

## 🏗️ Arquitetura do Código

```python
# Fluxo Principal
main() → carregar_dados() → treinar_modelo() → salvar_resultados()
```

### 🔧 Componentes Core

| Função | Responsabilidade |
|--------|------------------|
| `carregar_dados()` | Lê e prepara o dataset |
| `treinar_modelo()` | Treina e valida o modelo |
| `salvar_resultados()` | Armazena métricas em arquivo |

---

## 📁 Estrutura do Repositório

**Arquivos e Pastas:**
```
📂 TrabalhoAprendizadoDeMaquina/
├── 📄 modelo_de_regressao_linear.py
├── 📄 aerogerador.dat
└── 📂 teste_de_resultados_do_modelo/
    └── 📄 resultados_metricas.txt
```

---

## 👤 Autor & Licença

**Lucas Abreu** (@lucasabreuzip)  
🐙 [GitHub](https://github.com/lucasabreuzip) • 💼 [LinkedIn](https://www.linkedin.com/in/lucasabreuzip/)

📄 **Licença:** MIT License

---
> Este trabalho foi desenvolvido como parte de um projeto acadêmico para explorar conceitos de regressão linear e validação cruzada.

⭐ **Se este projeto te ajudou, dê uma estrela!** ⭐