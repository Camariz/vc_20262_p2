# VC 2026.2 — Prática II - Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet


## Objetivos

1. Construir um dataset de classificação de dígitos (10 classes) a partir das anotações do dataset WaterMeters.
2. Treinar AlexNet, VGG16 e ResNet18 com *transfer learning* e uma CNN pequena treinada do zero (`smallcnn`).
3. Avaliar e comparar os quatro modelos em desempenho e custo computacional.
4. Analisar criticamente os erros e as limitações do pipeline.

## Arquivos deste repositório

| Arquivo | Uso |
|---|---|
| `README.md` | Este roteiro |
| `requirements.txt` | Dependências |
| `splits.csv` | Divisão **oficial** treino/validação/teste (por imagem). Não altere |
| `README_ENTREGA.md` | Modelo do README que o grupo deve entregar |
| `.gitignore` | Exclui dados, recortes, checkpoints e credenciais |

## Regras

- **Licença do dataset: CC BY-NC-ND 4.0.** Uso não comercial e restrito à disciplina. É proibido versionar imagens do dataset, recortes gerados ou checkpoints. Antes do *commit*, limpe a saída das células que exibem imagens do dataset.
- **Uso de IA:** permitido para apoio na codificação. A análise, as respostas e o README devem ser autorais. Declare no README onde houve apoio de IA.
- **Protocolo único:** os quatro modelos devem ser treinados e avaliados nas mesmas condições (seção *Protocolo de treino*).

---

## Parte 1 — Aquisição dos dados

Baixe o dataset [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) pela API do Kaggle e verifique a integridade do download.

**Questão 1.1** — O que contém cada pasta e cada coluna do `data.csv`? Quais pastas servem para treinar um modelo?

**Questão 1.2** — O rótulo disponível é por imagem ou por dígito? O que isso implica para a tarefa?

## Parte 2 — Análise exploratória

Explore as imagens, as máscaras e os rótulos. Caracterize a distribuição das leituras.

**Questão 2.1** — Qual é a fonte mais confiável para o rótulo de cada imagem? Justifique com evidências.

**Questão 2.2** — Qual dígito deve ser a classe majoritária? Por quê?

## Parte 3 — Construção do dataset de dígitos

Requisitos:

- Rótulo de cada imagem com **8 dígitos** (5 inteiros + 3 decimais). Descarte as imagens que não se encaixam nesse formato.
- Visor retificado em uma faixa horizontal de **384×64 px**.
- Faixa dividida em **8 recortes de 48×64 px**, um por dígito.
- Correção de visores na vertical e de cabeça para baixo. Reporte quantas faixas foram corrigidas.
- Divisão conforme o `splits.csv` oficial.
- Recortes salvos em `digits/{split}/{digito}/{image_id}_{posicao}.png` e índice em `digits/index.csv` com as colunas `image_id, photo_name, position, digit, split, crop_path`.

Verificações obrigatórias: **1243** imagens válidas, **1** descartada, **9944** recortes, nenhuma imagem presente em mais de um split.

**Questão 3.1** — Por que a divisão é feita por imagem e não por recorte? Por que todos os grupos usam o mesmo `splits.csv`?

**Questão 3.2** — Quantas faixas foram corrigidas? Identifique e quantifique os tipos de recorte errado que o pipeline produz.

**Questão 3.3** — Que erro a divisão da faixa em larguras iguais introduz? Como você o corrigiria?

## Parte 4 — Treinamento dos quatro modelos

### Protocolo de treino


| Modelo | Pesos | Entrada | Batch (CUDA) | Batch (MPS/CPU) | LR cabeça | LR backbone | Épocas |
|---|---|---|---|---|---|---|---|
| `smallcnn` | do zero | 64×48 | 256 | 128 | 1e-3 | — | 10 |
| `alexnet` | IMAGENET1K_V1 | 128×128 | 128 | 64 | 1e-3 | 1e-4 | 8 |
| `vgg16_bn` | IMAGENET1K_V1 | 128×128 | 32 | 16 | 1e-3 | 1e-5 | 8 |
| `resnet18` | IMAGENET1K_V1 | 128×128 | 128 | 64 | 1e-3 | 1e-4 | 8 |

Comum a todos:

- otimizador `AdamW` e agendador `OneCycleLR`;
- `CrossEntropyLoss(label_smoothing=0.1)`;
- 1 época inicial com o backbone congelado (modelos pré-treinados);
- *early stopping* pelo F1-macro da validação, com paciência 3;
- `seed=42`;
- *augmentation*: `RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` + `ColorJitter(brightness=0.3, contrast=0.3)`, sem flip.

Alterações são permitidas desde que valham para os quatro modelos e estejam documentadas no README.

Salve o histórico de cada modelo em `history_{modelo}.csv` com as colunas `epoch, train_loss, train_acc, val_loss, val_acc, val_f1, seconds`.

**Questão 4.1** — Por que normalizar as imagens com a média e o desvio padrão do ImageNet?

**Questão 4.2** — Compare o número de parâmetros dos quatro modelos. Onde se concentram os parâmetros da AlexNet e da VGG16?

**Questão 4.3** — Qual a diferença entre treinar só a cabeça e fazer *fine-tuning* completo? Por que congelar o backbone apenas na primeira época?

**Questão 4.4** — Por que não usar flip neste problema?

## Parte 5 — Avaliação

Todas as métricas são calculadas no **conjunto de teste**. A validação serve apenas para *early stopping*.

### `results.csv` (uma linha por modelo)

| Coluna | Definição |
|---|---|
| `model` | nome do modelo |
| `acc_digit` | acurácia por recorte |
| `f1_macro` | F1 macro sobre as 10 classes |
| `acc_reading` | fração de imagens com os 8 dígitos corretos |
| `params_M` | número de parâmetros, em milhões |
| `s_per_epoch` | mediana do tempo das épocas de treino, sem validação |
| `ms_per_image` | inferência com batch 1: mediana de 200 execuções após 20 de *warm-up* |
| `ckpt_MB` | tamanho em disco do melhor checkpoint |
| `best_epoch` | época com o maior F1-macro de validação |

### Figuras e relatórios

- `classification_report` de cada modelo
- `figures/confusion_matrices.png` — matrizes de confusão normalizadas
- `figures/accuracy_by_position.png` — acurácia por posição do dígito (0–7)
- `figures/training_curves.png` — curvas de loss e acurácia (treino × validação)
- `figures/performance_vs_cost.png` — F1-macro × tempo de inferência
- Galeria com os 16 erros mais confiantes de cada modelo (apenas no notebook, saída limpa antes do commit)

**Questão 5.1** — Qual modelo teve o melhor F1-macro? O ranking é o mesmo da acurácia? Por quê?

**Questão 5.2** — Quais pares de classes mais se confundem? Por quê?

**Questão 5.3** — Por que `acc_reading` é muito menor que `acc_digit`?

**Questão 5.4** — Qual modelo você colocaria em um aplicativo de celular? Justifique com números.

**Questão 5.5** — O que a ResNet traz em relação à VGG16 neste problema?

**Questão 5.6** — A `smallcnn` foi competitiva com as redes pré-treinadas? O que isso indica?

## Desafio extra (opcional, até +0,5 na nota)

Escolha um:

- Melhore a divisão da faixa em dígitos e meça o ganho com o mesmo protocolo.
- Substitua a classificação por dígito por um modelo de sequência que leia a faixa inteira e compare `acc_reading`.

---

## Entrega

**O entregável é obrigatoriamente o link para o repositório GitHub do grupo**, enviado no Google Classroom até a data informada.

- Um único link por grupo, apontando para o repositório (`https://github.com/<usuario>/<repositorio>`).
- O repositório deve estar acessível ao professor (público ou com o professor adicionado como colaborador).
- Entregas em qualquer outro formato (arquivo anexado, `.zip`, notebook avulso, link do Drive ou do Colab) **não serão aceitas**.
- Será avaliado o estado do repositório no prazo de entrega; commits posteriores não serão considerados.

O repositório deve conter:

- notebook executado (sem imagens do dataset nas saídas);
- `splits.csv` oficial, `results.csv`, `history_*.csv` e a pasta `figures/`;
- `README.md` preenchido a partir do `README_ENTREGA.md`: reprodução, protocolo, resultados, respostas às questões, limitações e declaração de uso de IA.

Não versionar: dataset, recortes, checkpoints, `kaggle.json`.

## Critérios de avaliação

| Item | Peso |
|---|---|
| Aquisição e preparação do dataset (inclui a correção de orientação) | 10% |
| Treinamento dos quatro modelos com protocolo idêntico | 25% |
| Avaliação completa com todas as métricas da Parte 5 | 30% |
| Análise crítica e respostas às questões | 35% |

### Descritores

**Protocolo idêntico (25%)**

| Nível | Descrição |
|---|---|
| Excelente | `splits.csv` oficial nos quatro modelos, seed fixa, hiperparâmetros documentados, execução reproduzível |
| Bom | Protocolo idêntico, sem documentação dos hiperparâmetros |
| Regular | Um modelo com protocolo diferente, sem justificativa |
| Insuficiente | Splits regerados ou recortes da mesma imagem em treino e teste |

**Avaliação (30%)**

| Nível | Descrição |
|---|---|
| Excelente | Todas as métricas no teste, figuras legíveis, `results.csv` coerente com o README |
| Bom | Uma métrica ou figura ausente |
| Regular | Várias métricas ou figuras ausentes |
| Insuficiente | Métricas ausentes ou relatadas sobre a validação |

**Análise (35%)**

| Nível | Descrição |
|---|---|
| Excelente | Respostas fundamentadas nos próprios números, limitações identificadas de forma autônoma, conclusão sobre custo × benefício |
| Bom | Respostas corretas, com referência parcial aos resultados |
| Regular | Respostas parcialmente corretas ou pouco fundamentadas |
| Insuficiente | Respostas genéricas, sem referência aos resultados obtidos |

### Penalidades

| Situação | Penalidade |
|---|---|
| Entrega que não seja o link de um repositório GitHub acessível | Entrega não aceita |
| Vazamento entre splits | Zera o item de protocolo (25%) |
| Uso de modelos sem ser os citados | −30% da nota final |
| Imagens do dataset, recortes ou checkpoints versionados | −10% da nota final |
| Ausência do Roteiro de Uso de IA | −20% da nota final |

---

**Fonte dos dados:** Yandex.Toloka Water Meters Dataset, disponível no Kaggle sob licença CC BY-NC-ND 4.0.
- https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset?resource=download

# [Nome do grupo] — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

> **Modelo de README.** Copie este arquivo para o `README.md` do repositório do grupo e preencha
> todas as seções. Apague as linhas de instrução (as citações iniciadas por `>`) antes de entregar.
> A entrega é **obrigatoriamente o link deste repositório GitHub**, enviado no Google Classroom.

**Disciplina:** Visão Computacional (VC001) — CESAR School, 2026.2
**Integrantes:** Nome 1 (e-mail) · Nome 2 (e-mail) · Nome 3 (e-mail)

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais.

## Como reproduzir

O projeto foi desenvolvido em duas etapas: preparação dos dados em ambiente local e treinamento dos modelos no Google Colab.

A aquisição, a exploração e o pré-processamento do dataset foram realizados no Visual Studio Code, em ambiente Windows, utilizando Python 3.11.9. As dependências foram instaladas em um ambiente virtual Python, conforme o arquivo `requirements.txt`.

O dataset Yandex.Toloka Water Meters foi obtido por meio da API do Kaggle. Após o download, verificou-se a integridade do arquivo compactado e realizou-se a extração das fotografias e anotações para a pasta `data/WaterMeters/`.

A preparação foi desenvolvida no notebook `pratica_watermeters.ipynb`. Foram utilizadas as anotações dos visores para extrair e retificar as regiões de leitura em imagens de 384 × 64 pixels. Posteriormente, cada faixa foi dividida em oito recortes de 48 × 64 pixels, seguindo o protocolo da atividade. O processamento resultou em 9.944 recortes, organizados por conjunto de dados e classe, com seus respectivos registros em `digits/index.csv`.

A divisão entre treinamento, validação e teste seguiu exclusivamente o arquivo `splits.csv` fornecido pelo professor, sem alterações, contemplando 870 fotografias para treinamento, 186 para validação e 187 para teste.

Devido às limitações de processamento do computador local, o treinamento foi realizado no Google Colab, com aceleração por GPU CUDA. Os recortes preparados foram transferidos por meio do Google Drive e utilizados pelo script `treinar_watermeters.py` para treinar e avaliar os modelos SmallCNN, AlexNet, VGG16-BN e ResNet18.

Para reproduzir o experimento, é necessário instalar as dependências indicadas em `requirements.txt`, obter o dataset original, executar o processamento das imagens e disponibilizar os recortes gerados ao script de treinamento. No ambiente Colab, a execução foi realizada com o seguinte comando:

```bash
python -u treinar_watermeters.py \
  --index /content/watermeters/digits/index.csv \
  --out /content/drive/MyDrive/WaterMeters_VC/resultados \
  --models smallcnn alexnet vgg16_bn resnet18 \
  --workers 2
```

Os resultados foram armazenados em arquivos CSV, acompanhados dos históricos de treinamento, relatórios de classificação e gráficos comparativos.

Ambiente utilizado: Visual Studio Code no Windows 11, Python 3.11.9 e Google Colab com aceleração CUDA.

Tempo de execução: os tempos individuais de treinamento e inferência estão registrados no arquivo `results.csv` e nos históricos de cada modelo. O tempo total, incluindo preparação e transferência dos dados, não foi cronometrado.

Divisão dos dados: `splits.csv` oficial, sem modificações.

Limitação do pré-processamento: embora os 9.944 recortes tenham sido gerados, a validação visual das orientações e da correspondência entre os recortes e seus rótulos não foi concluída para todas as fotografias. Essa limitação deve ser considerada na interpretação dos resultados.

## Protocolo de treino

Foram avaliadas quatro arquiteturas de redes neurais convolucionais: SmallCNN, AlexNet, VGG16-BN e ResNet18. A SmallCNN foi treinada do zero, enquanto as demais utilizaram pesos pré-treinados no ImageNet (IMAGENET1K_V1), seguindo a estratégia de *transfer learning*.

O treinamento foi realizado no Google Colab, com aceleração CUDA, utilizando a divisão oficial dos dados em treinamento, validação e teste.

| Modelo | Pesos | Entrada | Batch | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | Do zero | 64×48 | 256 | 1e-3 | — | 10 | 1 |
| alexnet | IMAGENET1K_V1 | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 8 |
| vgg16_bn | IMAGENET1K_V1 | 128×128 | 32 | 1e-3 | 1e-5 | 8 | 7 |
| resnet18 | IMAGENET1K_V1 | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 8 |

O protocolo estabelecido para o experimento prevê o otimizador AdamW, o agendador de taxa de aprendizagem OneCycleLR e a função de perda CrossEntropyLoss, com suavização de rótulos (label smoothing) de 0,1. Para os modelos pré-treinados, foi prevista uma época inicial com o backbone congelado, seguida de ajuste fino dos parâmetros.

A estratégia de aumento de dados definida no roteiro compreende transformações geométricas leves com RandomAffine, incluindo rotação de até 5°, translação de até 5% e escala entre 0,95 e 1,05, além de ajustes de brilho e contraste com ColorJitter. Não foram previstas inversões horizontais ou verticais.

O treinamento foi configurado com semente aleatória 42 e critério de parada antecipada (early stopping) baseado no F1-macro do conjunto de validação, com paciência de três épocas. A avaliação final foi realizada exclusivamente no conjunto de teste.

Alterações em relação ao protocolo oficial: foram utilizados os tamanhos de lote definidos para CUDA, sem alteração dos limites máximos de épocas. A SmallCNN foi interrompida antecipadamente após quatro épocas, tendo alcançado seu melhor F1-macro de validação na primeira. Os outros três modelos completaram oito épocas. A execução das demais configurações deve ser conferida no script `treinar_watermeters.py`, que documenta os parâmetros efetivamente implementados.

## Resultados (conjunto de teste)

Os quatro modelos foram avaliados no conjunto de teste definido pelo arquivo `splits.csv` oficial, composto por 187 fotografias e 1.496 recortes de dígitos.

A avaliação considerou a acurácia por dígito (*acc_digit*), o F1-macro, a acurácia da leitura completa (*acc_reading*) e indicadores de custo computacional.

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| smallcnn | 0.3442513368983957 | 0.051218299353555444 | 0.0 | 0.48929 | 5.412428542000043 | 0.42308249999223335 | 1.8708696365356445 | 1 |
| alexnet | 0.9064171122994652 | 0.8880246791723623 | 0.5935828877005348 | 57.04481 | 13.007596166999974 | 2.251298999908613 | 217.61430644989014 | 8 |
| vgg16_bn | 0.9224598930481284 | 0.909353470922969 | 0.7058823529411765 | 134.309962 | 48.94256794900002 | 7.813363999957801 | 512.4149541854858 | 7 |
| resnet18 | 0.9378342245989305 | 0.9325208619185148 | 0.7593582887700535 | 11.181642 | 13.8592079094999 | 2.652494500125613 | 42.73029613494873 | 8 |

A ResNet18 apresentou o melhor desempenho geral, alcançando acurácia de 93,78% por dígito, F1-macro de 93,25% e acurácia de 75,94% na leitura completa. Além disso, apresentou menor número de parâmetros e menor tempo de inferência que a VGG16-BN.

A VGG16-BN obteve o segundo melhor desempenho, seguida pela AlexNet. A SmallCNN apresentou desempenho significativamente inferior, com acurácia de 34,43% e F1-macro de 5,12%, indicando dificuldades na classificação equilibrada das dez classes.

A diferença entre a acurácia por dígito e a acurácia da leitura completa evidencia a maior exigência desta última métrica, que considera correta somente a fotografia cujos oito dígitos foram classificados adequadamente.

Esses resultados devem ser interpretados considerando as limitações identificadas na preparação do dataset, especialmente as orientações não verificadas e os possíveis desalinhamentos entre recortes e rótulos.

### Figuras

As visualizações utilizadas na avaliação estão disponíveis na pasta `figures/`:

- **Matrizes de confusão normalizadas:** `figures/confusion_matrices.png`
- **Acurácia por posição do dígito:** `figures/accuracy_by_position.png`
- **Curvas de treinamento e validação:** `figures/training_curves.png`
- **Relação entre desempenho e custo computacional:** `figures/performance_vs_cost.png`

Os relatórios de classificação e as demais informações complementares estão disponíveis na pasta `reports/`.

## Respostas às questões

> Responda com base nos **seus** números. Respostas genéricas, sem referência aos resultados, são avaliadas como insuficientes.

**1.1** : O dataset WaterMeters está organizado em três pastas principais: images, masks e collage. A pasta images contém as fotografias originais dos hidrômetros, utilizadas como fonte para a extração dos dígitos. A pasta masks contém as máscaras associadas às regiões anotadas, enquanto collage reúne representações visuais auxiliares das anotações. O arquivo data.csv possui três colunas: photo_name, que identifica o arquivo da fotografia; value, que registra a leitura numérica do hidrômetro; e location, que contém a anotação espacial do visor, representada por coordenadas normalizadas de um polígono. Na exploração, identificamos 1.244 fotografias, sem valores ausentes nessas três colunas. A pasta images foi a principal fonte para construir o conjunto de treinamento, enquanto as anotações de data.csv foram utilizadas para localizar e retificar os visores. As pastas masks e collage podem auxiliar na inspeção e validação visual das regiões anotadas, mas não foram utilizadas diretamente como imagens de entrada dos modelos.

**1.2**: O dataset fornece o rótulo por fotografia, representando a leitura completa do hidrômetro, e não rótulos individuais para cada algarismo. Portanto, foi necessário transformar a leitura em uma sequência padronizada de oito dígitos, composta por cinco posições inteiras e três decimais. Em seguida, cada região anotada do visor foi retificada para 384 × 64 pixels e dividida em oito recortes de 48 × 64 pixels. Cada recorte recebeu como rótulo o algarismo correspondente à sua posição na sequência. Das 1.244 fotografias originais, 1.243 atenderam ao formato adotado, resultando em 9.944 recortes. Uma fotografia foi descartada por apresentar quatro casas decimais no nome do arquivo. Esse procedimento permitiu construir um dataset para classificação em dez classes, de 0 a 9. Entretanto, também introduziu possíveis erros de correspondência entre os recortes e os rótulos, especialmente quando os dígitos não estavam igualmente espaçados ou quando o visor apresentava menos de oito posições visíveis.

**2.1**: A fonte mais confiável para a obtenção inicial dos rótulos foi o nome dos arquivos das fotografias, pois apresenta a leitura do hidrômetro de maneira estruturada, permitindo identificar separadamente a parte inteira e a decimal. O campo value, disponível no arquivo data.csv, foi utilizado como referência para verificar a consistência dessas informações. Durante a análise exploratória, constatamos que os valores extraídos dos nomes eram numericamente compatíveis com os registrados em data.csv. Das 1.244 fotografias, 1.243 apresentaram um formato compatível com a padronização em oito dígitos. Apenas a fotografia id_664_value_45_2676.jpg foi descartada por apresentar quatro casas decimais. Entretanto, essa correspondência numérica não garante que todos os dígitos estejam corretamente representados nos recortes. A inspeção visual identificou casos em que o visor apresentava cinco posições visíveis, apesar de o rótulo padronizado possuir oito dígitos. Portanto, os nomes dos arquivos e o data.csv são fontes consistentes para a leitura numérica, mas a validação visual é necessária para confirmar a correspondência entre cada recorte e seu respectivo rótulo.

**2.2**: O dígito 0 deve ser a classe majoritária, principalmente porque as leituras foram padronizadas em oito posições, sendo cinco destinadas à parte inteira e três à parte decimal. Como muitos hidrômetros apresentam valores inferiores a 10.000, a representação exige o preenchimento das posições iniciais com zeros à esquerda. Essa predominância foi confirmada na análise exploratória. Dos 9.944 rótulos individuais gerados, 3.549 correspondem ao dígito zero, representando aproximadamente 35,69% do total. A distribuição também evidenciou maior concentração de zeros nas primeiras posições: aproximadamente 99,76% dos rótulos da posição 0 e 95,90% da posição 1 correspondiam a essa classe. Esse desequilíbrio pode influenciar o treinamento dos classificadores, favorecendo previsões da classe mais frequente. Por isso, a avaliação não deve considerar apenas a acurácia global, mas também o F1-macro, que atribui o mesmo peso às dez classes.

**3.1**: A divisão por fotografia evita vazamento de dados, pois recortes de uma mesma imagem compartilham características de iluminação, textura e enquadramento. Caso fossem distribuídos entre treino e teste, poderiam produzir resultados artificialmente elevados. O uso do splits.csv oficial garante condições equivalentes de avaliação entre os grupos. Foram utilizadas 1.243 imagens, distribuídas em 870 para treino, 186 para validação e 187 para teste, preservando os oito recortes de cada fotografia no mesmo conjunto.

**3.2**: Foram confirmadas e corrigidas 10 faixas invertidas, entre 25 fotografias inspecionadas visualmente. As outras 15 estavam corretamente orientadas, enquanto 1.218 permaneceram sem validação individual. Também foram identificadas 33 fotografias com cinco posições numéricas aparentes, entre 36 examinadas especificamente quanto a essa característica. Os problemas observados incluíram cortes parciais de algarismos, presença de dígitos vizinhos e incompatibilidade entre os recortes e seus rótulos. Esses últimos erros não foram quantificados individualmente em todo o dataset. 

**3.3**: A divisão em oito regiões de 48 × 64 pixels pressupõe espaçamento uniforme entre os algarismos. Quando essa condição não é atendida, os recortes podem conter dígitos incompletos ou partes de números vizinhos, comprometendo a correspondência com os rótulos. Uma melhoria seria localizar individualmente os algarismos por projeções verticais ou técnicas de detecção, realizando os recortes conforme suas posições reais. Visores com menos de oito dígitos visíveis exigiriam tratamento específico.

**4.1**: A normalização com a média e o desvio padrão do ImageNet mantém as entradas na escala utilizada durante o pré-treinamento da AlexNet, VGG16-BN e ResNet18. Isso favorece o aproveitamento das características já aprendidas pelas redes e contribui para a estabilidade do treinamento, especialmente durante o fine-tuning.

**4.2**: A SmallCNN apresentou aproximadamente 0,49 milhão de parâmetros, seguida pela ResNet18 (11,18 milhões), AlexNet (57,04 milhões) e VGG16-BN (134,31 milhões). Na AlexNet e na VGG16-BN, grande parte dos parâmetros concentra-se nas camadas totalmente conectadas do classificador. Apesar de possuir menos parâmetros que essas duas arquiteturas, a ResNet18 alcançou o maior F1-macro (93,25%), indicando melhor relação entre complexidade e desempenho neste experimento.

**4.3**: Treinar somente a cabeça significa atualizar apenas as camadas responsáveis pela classificação, preservando as características extraídas pelo backbone pré-treinado. No fine-tuning completo, os parâmetros do backbone também são ajustados para as características do novo dataset. O congelamento na primeira época permite adaptar inicialmente o classificador sem modificar as representações pré-treinadas. Posteriormente, o descongelamento possibilita especializar a rede no reconhecimento dos dígitos dos hidrômetros.

**4.4**: O flip horizontal ou vertical pode alterar a orientação, a posição e a interpretação visual dos algarismos, produzindo exemplos incompatíveis com seus rótulos. Como o reconhecimento depende da forma correta de cada dígito, essas transformações podem introduzir ruído no treinamento. Por esse motivo, foram previstas apenas pequenas alterações geométricas e de iluminação, preservando a orientação original dos números.

**5.1**: A ResNet18 apresentou o melhor F1-macro (93,25%), seguida pela VGG16-BN (90,94%), AlexNet (88,80%) e SmallCNN (5,12%). O ranking foi o mesmo para a acurácia por dígito, com valores de 93,78%, 92,25%, 90,64% e 34,43%, respectivamente. Essa concordância indica que os modelos com maior acurácia também apresentaram melhor desempenho entre as diferentes classes. Entretanto, o F1-macro é especialmente relevante diante do desequilíbrio dos dados, pois atribui o mesmo peso a cada classe.

**5.2** : A SmallCNN apresentou forte concentração das previsões na classe zero, confundindo praticamente todos os demais algarismos com essa classe. Nas redes pré-treinadas, os erros foram menos frequentes e distribuídos entre diferentes dígitos, com algumas confusões envolvendo as classes 0, 3, 4, 6 e 8. Essas dificuldades podem estar relacionadas à semelhança visual dos algarismos, aos reflexos e à presença de números parcialmente visíveis nos contadores mecânicos. A ResNet18 apresentou a maior concentração de acertos na diagonal da matriz, coerente com seu F1-macro de 93,25%.

**5.3**: A acurácia por dígito considera cada recorte individualmente, enquanto a acurácia da leitura completa exige que os oito algarismos de uma fotografia sejam classificados corretamente. Assim, um único erro torna toda a leitura incorreta. Na ResNet18, por exemplo, a acurácia por dígito foi de 93,78%, mas a acurácia da leitura completa caiu para 75,94%. Isso demonstra como erros individuais se acumulam na interpretação de uma sequência.

**5.4**: A ResNet18 seria a escolha mais adequada entre os modelos avaliados, considerando o equilíbrio entre desempenho e custo computacional. Alcançou F1-macro de 93,25%, acurácia por dígito de 93,78% e acurácia da leitura completa de 75,94%, utilizando aproximadamente 11,18 milhões de parâmetros. Seu tempo de inferência foi de 2,65 ms por recorte, contra 7,81 ms da VGG16-BN. Embora esses tempos tenham sido medidos na GPU do Colab, e não em um celular, os resultados indicam maior potencial de aplicação em dispositivos com recursos limitados.

**5.5**: A principal vantagem da ResNet18 está nas conexões residuais, que facilitam o fluxo de gradientes e o treinamento da rede. Neste experimento, a arquitetura apresentou melhor desempenho utilizando menos recursos: F1-macro de 93,25%, contra 90,94% da VGG16-BN, com aproximadamente 11,18 milhões de parâmetros, diante de 134,31 milhões. Além disso, apresentou inferência cerca de três vezes mais rápida, indicando melhor relação entre desempenho e custo computacional.

**5.6**: A SmallCNN não apresentou desempenho competitivo. Obteve acurácia por dígito de 34,43%, F1-macro de 5,12% e nenhuma leitura completa correta. Em comparação, as três redes pré-treinadas ultrapassaram 88% de F1-macro. A diferença evidencia as vantagens do aprendizado por transferência neste experimento, especialmente pela capacidade de aproveitar representações visuais previamente aprendidas. O baixo F1-macro da SmallCNN também sugere dificuldade em distinguir as classes menos frequentes, hipótese que deve ser confirmada pela matriz de confusão.

## Limitações

> O que o pipeline faz de errado e por quê? Quanto do erro restante vocês atribuem aos dados (rótulo, recorte)
> e quanto ao modelo? O que vocês mudariam?

A principal limitação do pipeline está na preparação das imagens. Embora tenham sido gerados 9.944 recortes, 1.218 fotografias permaneceram sem validação individual de orientação. Também foram identificados visores com menos de oito dígitos visíveis e recortes desalinhados, que podem comprometer a correspondência entre imagens e rótulos.

Os erros não podem ser atribuídos exclusivamente aos dados ou aos modelos, pois não foi realizada uma análise que quantificasse separadamente essas contribuições. Entretanto, a diferença entre o F1-macro da SmallCNN (5,12%) e da ResNet18 (93,25%) evidencia a influência da arquitetura e do aprendizado por transferência.

Como melhorias, propomos automatizar a correção da orientação, adotar uma segmentação adaptativa dos dígitos e revisar os rótulos inconsistentes, avaliando posteriormente o impacto dessas alterações no desempenho dos modelos.

## Declaração de uso de IA

> Seguir o template disponivel no Google Classroom da disciplina.

Foi utilizado o ChatGPT (OpenAI, GPT-6) como ferramenta de apoio
à implementação, depuração, interpretação dos resultados e revisão
textual da documentação.

A declaração detalhada, incluindo exemplos literais dos prompts,
contribuições incorporadas e procedimentos de validação, está
disponível em [ROTEIRO_USO_IA.md](ROTEIRO_USO_IA.md).


