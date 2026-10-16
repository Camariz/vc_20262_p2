# DECLARAÇÃO DE USO DE INTELIGÊNCIA ARTIFICIAL

## Identificação da atividade

**Disciplina:** Visão Computacional — VC001  
**Instituição:** CESAR School  
**Período:** 2026.2  
**Atividade:** Prática II — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet18  
**Data:** 15/10/2026  
**Aluno(a):** Carolline de Araújo Mariz
**Matrícula:** 

## SEÇÃO A — Uso de LLM / Assistente de IA

**Utilizou IA?** Sim.

**Ferramenta e modelo utilizado:** ChatGPT — OpenAI, GPT-6.

### 1. Prompts utilizados

A ferramenta foi utilizada como apoio ao desenvolvimento do projeto, por meio de solicitações relacionadas às seguintes atividades:

- Preparação dos dados: orientações para aquisição, exploração e organização do dataset WaterMeters, padronização dos rótulos e utilização da divisão oficial entre treinamento, validação e teste.
- Implementação e depuração: auxílio na elaboração e correção de funções Python para retificação dos visores, segmentação dos algarismos, geração dos recortes e organização dos arquivos.
- Treinamento:orientações para configuração do ambiente, utilização do Google Colab e execução das arquiteturas SmallCNN, AlexNet, VGG16-BN e ResNet18.
- Avaliação: apoio à interpretação das métricas, matrizes de confusão, comparação entre modelos e identificação de limitações do processamento.
- Documentação: auxílio na estruturação, revisão e síntese dos textos acadêmicos e das informações apresentadas no README.

Como exemplos de solicitações realizadas durante o desenvolvimento, apresentam-se os seguintes prompts transcritos literalmente da conversa:

Exemplo 1 — Ambiente de treinamento

> mas vamos rodar no vs code mesmo? e não no colab?

Exemplo 2 — Documentação

> Preciso que vc me de um texto adequado a realidade que fizemos. Não irei modificar nada. Por favor seja direto academico e utilize uma linguagem humana.

Exemplo 3 — Revisão das respostas

> Por favor, preciso da sua ajuda para sintetizar as minhas respostas. Por favor, mantenha o conteúdo que precisa ser apresentado.

**Exemplo 4 — Declaração de IA**

> Aqui esta o modelo para uso de IA. Podes me ajudar?

Os tópicos anteriores sintetizam as demais solicitações realizadas. Os exemplos literais não constituem um registro completo de todos os prompts da atividade.

### 2. Respostas geradas e incorporadas ao trabalho

O ChatGPT forneceu orientações, explicações e sugestões de código utilizadas na construção do pipeline de preparação dos dados, incluindo as funções de padronização dos rótulos, retificação dos visores, segmentação dos algarismos e organização dos recortes.

A ferramenta também contribuiu para a elaboração do script `treinar_watermeters.py`, utilizado no treinamento e na avaliação das quatro arquiteturas no Google Colab.

Na documentação, houve apoio na organização do protocolo experimental, apresentação dos resultados, interpretação das métricas e revisão textual das respostas às questões propostas.

Os principais trechos de implementação apoiados por IA estão presentes no notebook `pratica_watermeters.ipynb` e no script `treinar_watermeters.py`.

### 3. Adaptações e validações realizadas

As sugestões de código foram executadas nos ambientes Visual Studio Code e Google Colab, com verificações relacionadas à estrutura dos dados, dimensões dos recortes, integridade dos arquivos e funcionamento do treinamento.

O processamento resultou em 1.243 fotografias válidas e 9.944 recortes. Durante a inspeção, foram identificadas limitações relacionadas à orientação de alguns visores e à correspondência entre as posições visíveis e os rótulos padronizados. Essas ocorrências foram registradas e consideradas na discussão das limitações do projeto.

Os quatro modelos foram treinados e avaliados, e as métricas apresentadas na documentação foram confrontadas com os resultados obtidos e com o arquivo `results.csv`.

A utilização das sugestões de IA não elimina as limitações identificadas, nem substitui a responsabilidade dos integrantes pela compreensão, revisão e interpretação dos resultados.

## SEÇÃO B — Uso de Skill (Claude Code / SKILL.md)

Utilizou skill? Não, considerando que não foram empregadas ferramentas específicas baseadas em arquivos SKILL.md no desenvolvimento relatado.

## SEÇÃO C — Uso de SDD (System Design Document)

Utilizou SDD? Não, considerando que não foi empregado um documento formal de especificação de sistema no desenvolvimento relatado.

## DECLARAÇÃO DE RESPONSABILIDADE

Declaro estar ciente de que ferramentas de inteligência artificial foram utilizadas como apoio ao desenvolvimento, à implementação e à documentação desta atividade.

Reconheço a responsabilidade pela compreensão, revisão e correção do conteúdo submetido, incluindo os trechos desenvolvidos com auxílio de IA, bem como a necessidade de apresentar os resultados e as limitações do experimento de forma transparente.

**Aluno(a):** Carolline de Araújo Mariz 
**Data:** 15/10/2026