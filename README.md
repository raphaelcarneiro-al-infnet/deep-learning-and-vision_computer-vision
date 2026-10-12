# Visão Computacional com CNNs e Transformers — Projeto da Disciplina

**Autor:** Raphael Alves Carneiro
**Instituição:** Instituto Infnet — Pós-graduação

Projeto em quatro atividades que cobrem o arco da disciplina: um Vision Transformer construído do zero e comparado com fine-tuning, busca semântica com CLIP em publicidade, transfer learning com CNN e um estudo de caso de triagem de COVID-19 em radiografias com augmentation sintética por GAN condicional. O item 4.2 (tráfego urbano) é entregue só no relatório.

Todos os notebooks rodam do início ao fim no **Google Colab com GPU T4**, sem modificações além de montar o Google Drive. Os notebooks estão salvos **com as saídas da execução**, então dá para ler resultados e gráficos sem rodar nada.

## Notebooks

| Atividade | Notebook | Abrir no Colab | Tempo na T4 |
|---|---|---|---|
| A1 — Vision Transformers (catálogo de moda) | `A1_vision_transformers.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelcarneiro-al-infnet/deep-learning-and-vision_computer-vision/blob/main/A1_vision_transformers.ipynb) | ~20–25 min |
| A2 — CLIP no ADS-16 | `A2_clip_ads16.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelcarneiro-al-infnet/deep-learning-and-vision_computer-vision/blob/main/A2_clip_ads16.ipynb) | ~10–15 min |
| A3 — CNN pré-treinada (feature extraction) | `A3_cnn_kaggle.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelcarneiro-al-infnet/deep-learning-and-vision_computer-vision/blob/main/A3_cnn_kaggle.ipynb) | ~10–15 min |
| A4.1 — Raio-X COVID-19 + GAN condicional | `A4_estudo_caso_raio_x.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelcarneiro-al-infnet/deep-learning-and-vision_computer-vision/blob/main/A4_estudo_caso_raio_x.ipynb) | ~55–60 min |

## Resumo das atividades e principais resultados

### A1 — Vision Transformers em classificação de produtos de e-commerce
Classificação de 10 tipos de produto do *Fashion Product Images* (Hugging Face `ashraq/fashion-product-images-small`), um problema real de tagueamento de catálogo para marketplaces e feeds de anúncios.
- Scaled dot-product attention e multi-head attention **implementados do zero**, com projeções independentes por head e 9 testes unitários (incluindo equivalência numérica com `nn.MultiheadAttention`).
- Encoder block, patch embedding, CLS token e positional encoding próprios, mais um experimento que mostra que, **sem PE, o ViT dá a mesma resposta para a imagem com os patches embaralhados**.
- **ViT do zero** (1,8 M parâmetros, 64 px): 89,9% de accuracy no teste. **ViT-Base pré-treinado com fine-tuning** (85,8 M, 224 px): **95,6%**.
- Attention maps por head, attention rollout, distância média de atenção por camada e fração da atenção no fundo branco, com leitura do artefato de "registers" no ViT-Base.

### A2 — Reconhecimento semântico em publicidade visual com CLIP
CLIP ViT-L/14 no ADS-16, sem nenhum treino: 1.547 imagens após deduplicação (299 anúncios em 20 categorias + 1.248 imagens do corpus dos participantes).
- Ranking de 25 conceitos com limiar justificado (probabilidade zero-shot ≥ 2,5× o acaso) e análise de sensibilidade. Nos anúncios dominam texto/logo (65%) e embalagem de produto (45%).
- Sem rótulos, o CLIP recupera o tema de várias categorias (carros 92%, joias/relógios 93%, pets 87%).
- 10 consultas textuais do concreto ao abstrato: o CLIP **lê o texto da imagem** e tem **composicionalidade fraca** ("a cold beer" traz paisagens de neve).
- Experimento de padding e attention mask: sem a mask, o vetor do BERT muda (cosseno 0,71); o do CLIP não muda (1,00).

### A3 — Classificador com CNN pré-treinada
ResNet-50 (ImageNet) com backbone congelado e novo head, no dataset Kaggle `pavansanagapati/images-dataset` (7 classes, 1.764 imagens únicas após remover 1.842 duplicatas).
- 100% de accuracy no teste (265 imagens), com leitura estatística do resultado (limite superior de ~1,1% de erro pela regra do três).
- Propostas de augmentation justificadas por classe e experimento com vs. sem augmentation.

### A4.1 — Triagem de COVID-19 em radiografias com GAN condicional
Reprodução do cenário do enunciado (840 / 240 / 120 imagens) a partir do *COVID-19 Radiography Database*, com teste externo de 2.000 imagens.
- Diagnóstico de 8 problemas técnicos do projeto original, com impacto clínico. A **réplica da receita original** reproduz o problema: 88,6% de accuracy e só **26% de recall de COVID** (140 de 200 casos classificados como Normal).
- **GAN condicional em duas versões:** a DCGAN clássica (v1) sofre **mode collapse completo**; a versão estabilizada (v2: hinge loss, spectral norm, DiffAugment, TTUR e EMA) reduz o KID de COVID de 590 para 245 e recupera a diversidade (0,04 → 0,71). Checagem de memorização sem sinal de cópia do treino.
- **Recall de COVID** (média de 3 sementes): receita corrigida só com imagens reais, 82,2%; com reamostragem, 80,7%; com 400 COVID sintéticas, 83,7%. O ganho da GAN (+1,5 ponto) fica dentro da variação entre sementes e **não é adotado** pelo critério definido no plano. **Quase todo o ganho (26% → 82%) vem de corrigir a receita.**
- Plano de melhoria integrado com critério de adoção clínica.

## Como executar

1. Clique no botão **Open in Colab** do notebook desejado.
2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, escolha **GPU T4**.
3. **Ambiente de execução → Executar tudo**, e autorize o Google Drive quando o Colab pedir. As figuras e métricas são salvas em `MyDrive/PD_VisaoComputacional/`.

**Dados.** Os notebooks baixam os datasets sozinhos (Hugging Face ou `kagglehub`). Se o Kaggle exigir autenticação, basta deixar o arquivo `kaggle.json` (Kaggle → Settings → API → Create New Token) na raiz do Google Drive, em `MyDrive/kaggle.json`.

| Dataset | Fonte | Usado em |
|---|---|---|
| Fashion Product Images (small) | Hugging Face `ashraq/fashion-product-images-small` | A1 |
| ADS-16 Computational Advertising Dataset | Kaggle `groffo/ads16-dataset` | A2 |
| Images Dataset | Kaggle `pavansanagapati/images-dataset` | A3 |
| COVID-19 Radiography Database | Kaggle `tawsifurrahman/covid19-radiography-database` | A4.1 |

## Referências principais

- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.
- Touvron et al. (2021). *Training data-efficient image transformers & distillation through attention* (DeiT). ICML.
- Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
- Darcet et al. (2024). *Vision Transformers Need Registers.* ICLR.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Miyato & Koyama (2018). *cGANs with Projection Discriminator.* ICLR.
- Zhao et al. (2020). *Differentiable Augmentation for Data-Efficient GAN Training.* NeurIPS.
- DeGrave, Janizek & Lee (2021). *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence.
