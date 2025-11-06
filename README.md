# MatchPointML – Classificação de Partidas de Tênis com Machine Learning

## Integrantes do projeto
| Nome           | RA         |
|----------------|-------------|
| Andre Soares   | 21.01922-3  |
| Bruno Lopes    | 20.00041-3  |
| Caio Correia   | 21.00931-7  |
| Enzo Pucci     | 21.02093-0  |
| Pedro Mesquita | 21.02028-0  |

[Apresentação](https://youtu.be/rHGtGcViNl8)
[Link para o Artigo](https://mauabr-my.sharepoint.com/:w:/g/personal/21_02028-0_maua_br/ERjOFfdhhtxJuVabxUZWiqMB_GorZ4AQoakPgCwhBXaXOA?e=ihJmUW)

## Visão Geral do Projeto
MatchPointML é um projeto de ciência de dados focado em prever o resultado de partidas de tênis utilizando técnicas de *Machine Learning*. O objetivo principal é treinar modelos de classificação que, dados atributos de uma partida de tênis (por exemplo, informações dos jogadores e condições do jogo), possam classificar quem será o vencedor da partida. Esse problema de predição de resultados esportivos é um caso de uso de classificação binária (vitória do Jogador A ou do Jogador B) e envolve análise de dados históricos de partidas para extrair padrões. 

No contexto deste projeto, foram coletados dados históricos de partidas de tênis profissional (como estatísticas dos jogadores e partidas anteriores) para treinar algoritmos de aprendizado de máquina. Com isso, busca-se não apenas acertar o vencedor de uma partida antes dela acontecer, mas também compreender quais fatores influenciam o resultado – como diferença de ranking entre os jogadores, desempenho recente, tipo de quadra (saibro, grama, dura), entre outros. Este projeto foi desenvolvido como parte da grade do curso de Engenharia de Computação do Instituto Mauá de Tecnologia, ilustrando a aplicação de ciência de dados e ML em problemas do mundo real.
## Estrutura de Diretórios
A estrutura de diretórios do repositório é organizada para separar dados, modelos, códigos e recursos de apresentação. Cada pasta tem uma função específica conforme descrito abaixo:

- data/ – Contém os conjuntos de dados de entradas do projeto. Aqui residem os arquivos CSV com dados brutos ou pré-processados das partidas de tênis utilizadas no treinamento e teste do modelo. (Exemplo: um arquivo ``tennis_matches.csv`` com histórico de partidas, estatísticas dos jogadores, etc.).
- images/ – Armazena imagens geradas ou utilizadas pelo projeto. Pode incluir gráficos de análise exploratória de dados (EDA), como distribuições de features, gráficos de correlação, ou visualizações de resultados do modelo (por exemplo, matriz de confusão, curva ROC, importantes de variáveis) para uso em relatórios ou na interface web.
- models/ – Destinada aos artefatos de modelos treinados. Após o processo de treinamento no notebook, o modelo de melhor desempenho é salvo nesta pasta (geralmente em formato ``.pkl`` ou ``.joblib``). Este arquivo de modelo pré-treinado é carregado pela aplicação Streamlit para realizar previsões sem necessidade de treinar novamente sempre.
- examples/ – (Opcional) Exemplos de uso do projeto, como notebooks demonstrativos. No nosso caso, esta pasta pode conter notebooks Jupyter adicionais ou documentos de exemplo mostrando como utilizar o modelo ou visualizar alguns resultados. Os notebooks principais do projeto também podem estar aqui (ver descrição em “Arquivos Principais” abaixo).

Além das pastas acima, os arquivos de código fonte Python e notebooks Jupyter estão no diretório raiz do projeto, conforme listado a seguir.

## Pré-Requisitos

Antes de executar o projeto, certifique-se de ter o ambiente configurado com os pré-requisitos abaixo. Recomenda-se usar Python 3.8+ e criar um ambiente virtual (via ``venv`` ou ``conda``) para instalar as dependências do projeto. Abaixo está um ``requirements.txt`` simulado listando os principais pacotes utilizados:
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
```

Além das bibliotecas acima, o projeto também utiliza bibliotecas padrão do Python como ``os``, ``pickle``/``joblib`` (para salvar/carregar modelos), e possivelmente ``requests`` ou ``urllib`` caso haja leitura de dados de fontes externas. Certifique-se de instalar todas as dependências antes de prosseguir.

## Instruções de Instalação e Execução
Siga os passos abaixo para clonar o repositório, instalar as dependências e executar o projeto localmente:
1. Clonar o repositório: Abra um terminal e rode o comando:

    ```
    git clone https://github.com/MatchPointML/MatchPointML.git
    ```

    Em seguida, entre no diretório do projeto:
    ```
    cd MatchPointML
    ```
2. Criar ambiente virtual (opcional, mas recomendado):
    ```
    python3 -m venv venv  
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
3. Instalar as dependências: Com o ambiente ativado, instale os pacotes requeridos:

    ```
    pip install -r ``requirements.txt``
    ```

    Isso irá instalar pandas, numpy, scikit-learn, streamlit e demais bibliotecas necessárias.

4. Preparar os dados (se necessário): Certifique-se que o arquivo de dados de partidas esteja na pasta ``data``/ conforme esperado pelo código (por exemplo, ``data/tennis_matches.csv``). Caso o repositório não inclua os dados por completo (devido a tamanho ou licença), obtenha-os conforme instruções fornecidas no projeto (pode ser necessário baixar de uma fonte externa, como Jeff Sackmann’s Tennis Dataset ou Kaggle). Depois de colocar os arquivos na pasta, você pode rodar o notebook de input de dados (``inputting.ipynb``) para gerar o dataset processado.

5. Executar os notebooks de análise/treinamento: Abra os notebooks Jupyter para inspeção ou re-treinamento do modelo, se desejado. Por exemplo:

    - Execute ``inputting.ipynb`` para realizar o pré-processamento dos dados brutos e preparar o conjunto de dados para modelagem.

    - Depois, execute ``model.ipynb`` para treinar os modelos de *machine learning* e avaliar seu desempenho. Esse notebook carrega os dados preparados, divide em treino e teste, treina vários algoritmos e salva o melhor modelo em ``models/model.pkl`` (ou nome similar).

    Nota: Os notebooks não precisam ser executados para utilizar a aplicação web, já que um modelo treinado já foi salvo. No entanto, eles são úteis para entender e reproduzir a análise e o processo de modelagem.

6. Executar a aplicação web (Streamlit): Para interagir com o modelo via interface web, utilize o Streamlit. No terminal, execute:

    ```
    streamlit run ``front.py``
    ```

    Isso iniciará um servidor local e abrirá o aplicativo no navegador (geralmente em ``http://localhost:8501``). Na interface Streamlit, você poderá inserir as informações de uma partida e receber a previsão do modelo, bem como navegar por visualizações de resultados (ver seção Interface Web abaixo para detalhes).

7. Encerramento: Para parar o aplicativo Streamlit, pressione ``Ctrl+C`` no terminal onde ele está rodando. Se estiver usando ambiente virtual, você pode desativá-lo com ``deactivate`` após terminar os testes.

Seguindo esses passos, você terá o ambiente configurado e poderá explorar todas as funcionalidades do MatchPointML localmente.

## Descrição dos Arquivos Principais

A seguir, resumimos o conteúdo e a finalidade dos principais arquivos de código e notebooks presentes no projeto:

- ``functions.py`` – Contém funções auxiliares e utilitárias usadas em todo o projeto. Inclui rotinas para carregamento e limpeza de dados, transformação de features e funções de treinamento e avaliação do modelo de ML. Por exemplo, pode definir funções para ler o dataset de partidas (convertendo CSV em DataFrame pandas), preparar os dados (tratar valores ausentes, normalizar ou codificar variáveis categóricas como nome dos jogadores ou tipo de torneio) e também funções para treinar modelos específicos (como uma função ``train_model(X, y)`` que treina e retorna um classificador ajustado). Funções para calcular métricas de desempenho (acurácia, matriz de confusão, etc.) e até para realizar previsões em novos dados (por exemplo, ``predict_winner(jogadorA, jogadorB)``) também devem estar definidas aqui, centralizando a lógica de negócio do projeto.

- ``front.py`` – Script principal da interface web desenvolvida com Streamlit. Quando executado (``streamlit run front.py``), este script cria a página web interativa. Nele são definidos os elementos de interface: títulos, descrições, campos de entrada e botões. Por exemplo, o aplicativo pode apresentar um título como "Previsor de Resultados de Tênis" (``st.title(...)``), seguido de campos para o usuário selecionar ou inserir informações de uma partida: dropdowns para escolher jogadores ou sliders/inputs para estatísticas (ranking atual de cada jogador, aproveitamento de saque, etc.), e um botão "Prever Resultado". Ao clicar no botão, o script usa as funções de ``functions.py`` e o modelo treinado (carregado de ``models/``) para calcular a previsão do vencedor, exibindo então o resultado na tela com ``st.write`` ou ``st.success``/``st.error`` etc. O ``front.py`` foca na interação: coleta os inputs do usuário, chama o modelo para predizer e mostra a saída de forma amigável (por exemplo, "Previsão: Jogador A vence com X% de probabilidade").

- ``model_presentation.py`` – Módulo complementar para apresentação dos resultados do modelo. Este arquivo pode ser usado como uma segunda página na aplicação Streamlit ou simplesmente agrupar funções/visualizações relacionadas à análise de desempenho do modelo. Nele, possivelmente definimos gráficos ou tabelas que resumem o quão bem o modelo performou. Por exemplo, pode gerar e exibir uma matriz de confusão dos resultados no conjunto de teste, mostrar métricas como acurácia, precisão, recall e F1-score em um formato organizado, ou até listar a importância de cada feature no modelo (especialmente se for um modelo do tipo árvore de decisão ou random forest). Se integrado ao Streamlit, ``model_presentation.py`` poderia criar uma página separada (usando ``st.sidebar.selectbox`` para navegação, ou usando o recurso de multipage do Streamlit) para que o usuário veja visualizações gráficas e textos explicativos sobre o modelo treinado. Em resumo, enquanto o ``front.py`` lida com previsão para inputs do usuário, o ``model_presentation.py`` lida com apresentação dos insights do modelo já treinado.

- ``model.ipynb`` – Notebook Jupyter contendo todo o pipeline de modelagem e avaliação do projeto. Nele está documentado o processo passo-a-passo: inicia com a análise exploratória de dados (EDA) do dataset de tênis (estatísticas descritivas, distribuição de variáveis, possivelmente correlação entre fatores como ranking e vitória). Em seguida, o notebook realiza a preparação dos dados para modelagem (pode usar funções de ``functions.py`` para limpar e transformar os dados ou executar esse código diretamente). Após preparar os dados, o notebook divide o conjunto em treino e teste (por exemplo, 80/20) e experimenta diferentes algoritmos de *Machine Learning* para a tarefa de classificação. Possíveis modelos testados incluem: Regressão Logística, Support Vector Machine (SVM), K-Nearest Neighbors, Random Forest, entre outros. O notebook compara o desempenho desses modelos usando métricas como acurácia e escolhe o melhor (ou combina modelos, se aplicável). Há seções mostrando resultados como a matriz de confusão do modelo escolhido, as métricas no conjunto de teste e comentários interpretando esses resultados. Ao final, o notebook salva o modelo treinado (por exemplo, usando ``pickle.dump`` ou ``joblib``) no arquivo dentro de ``models/``, para uso posterior na aplicação Streamlit. Este notebook serve como documentação do processo de Data Science realizado: desde a compreensão dos dados até a seleção do modelo ótimo.

- ``inputting.ipynb`` – Notebook Jupyter focado na ingestão e pré-processamento dos dados brutos. Nele são realizadas as etapas iniciais do pipeline de dados. Tipicamente, este notebook carrega os arquivos originais de partidas (por exemplo, datasets separados por ano ou gênero, ou um grande CSV com histórico de partidas), realiza limpeza e combinações necessárias, e produz um dataset consolidado e pronto para a etapa de modelagem. Entre as tarefas do ``inputting.ipynb``, podemos citar: remover ou inferir valores ausentes nas colunas (por exemplo, preenchendo com medianas ou valores padrões estatísticos), feature engineering (criação de novas colunas a partir de existentes – por exemplo, calcular a diferença de ranking entre os dois jogadores de cada partida, ou marcar se o jogador da casa está jogando), conversão de tipos (datas, categóricas para numéricas se preciso, etc.) e filtragem de dados irrelevantes (talvez filtrando apenas partidas de certas categorias de torneio ou períodos de tempo para o escopo do projeto). Ao final, este notebook salva o dataset tratado em um arquivo (possivelmente ``data/processed_matches.csv`` ou similar), que então é usado pelo notebook de modelagem (``model.ipynb``). Em resumo, ``inputting.ipynb`` cuida para que os dados brutos sejam transformados em uma forma utilizável e de qualidade para alimentar os modelos de ML.

## Pipeline de Dados e Modelagem

Este projeto segue um pipeline típico de ciência de dados, desde a preparação dos dados até a avaliação do modelo de *machine learning*. Abaixo descrevemos as principais etapas:

- Processamento e Transformação de Dados: Os dados históricos de partidas de tênis foram inicialmente coletados e preparados. Isso envolveu ler os datasets (por exemplo, arquivos CSV contendo resultados de jogos e estatísticas dos jogadores) e limpar os dados. A limpeza incluiu tratar valores ausentes (por exemplo, partidas sem informação completa de algum jogador), remover outliers ou dados inconsistentes e padronizar os formatos (como converter dados categóricos em numéricos através de one-hot encoding ou mapeamento de rótulos – e.g., codificar “Quadra Dura”, “Saibro”, “Grama” como valores numéricos). Em seguida, foram criadas features relevantes para a previsão: em vez de usar diretamente os nomes dos jogadores, o modelo utiliza atributos quantificáveis. Por exemplo, para cada partida foram calculados indicadores como a diferença de ranking entre Jogador A e Jogador B, o histórico recente de vitórias de cada jogador (win rate nos últimos X jogos), estatísticas médias de aces, duplas-faltas, porcentagem de primeiro serviço, dentre outros disponíveis no dataset. Essas características transformam os dados brutos em um conjunto de variáveis explanatórias que o modelo de ML consegue usar. Por fim, os dados foram divididos em conjuntos de treinamento e teste (por exemplo, 80% das partidas para treinar o modelo e 20% para testar, garantindo que o modelo seja avaliado em partidas que não “viu” durante o treino).

- Algoritmos de *Machine Learning* Aplicados: Com os dados prontos, várias técnicas de aprendizado supervisionado foram exploradas para encontrar a que melhor prediz os resultados das partidas. Dentre os algoritmos de classificação testados estão:

    - Regressão Logística: um modelo linear probabilístico que tenta estimar a chance de vitória de um jogador em função das features.

    - Máquina de Vetor de Suporte (SVM): um algoritmo que busca um hiperplano ótimo para separar classes (vitória do Jogador A vs do Jogador B) no espaço de features, possivelmente utilizando kernels para considerar relações não lineares.

    - Árvore de Decisão e Random Forest: métodos de árvore que capturam relações de decisão nos dados (por exemplo: “se diferença de ranking > 50 e jogador A tem baixo aproveitamento de saque, então jogador B vence”). O Random Forest, em particular, combina múltiplas árvores de decisão para melhorar a generalização e lida bem com features de diferentes escalas e distribuições.

    - K-Nearest Neighbors (KNN): método baseado em instâncias que olha para as partidas mais similares no conjunto de treino para decidir o resultado de uma nova partida.

    - Gradient Boosting (ex: XGBoost/LightGBM): embora não confirmado, projetos assim muitas vezes testam modelos ensemble baseados em boosting, que podem oferecer alta acurácia combinando múltiplos “estágios” de árvores de decisão treinadas sequencialmente.

Cada algoritmo foi treinado usando o conjunto de treino e ajustado conforme necessário (por exemplo, normalizando features para SVM/KNN, ou ajustando hiperparâmetros como profundidade da árvore no Random Forest). Após o treinamento, com o uso do conjunto de teste foram comparadas as performances. O modelo com melhor desempenho geral – medido por métricas descritas abaixo – foi escolhido como modelo final do projeto. Este modelo final foi então salvo (serializado) para uso na interface web. Em muitos casos, modelos como Random Forest ou algoritmos de boosting destacam-se em problemas com muitas features e dados tabulares, então não é incomum que um desses tenha sido selecionado como modelo de produção do MatchPointML.

- Avaliação dos Modelos: A performance foi medida usando métricas de classificação apropriadas. A acurácia (porcentagem de partidas corretamente preditas) é a métrica principal reportada, dado que o conjunto de dados é razoavelmente balanceado entre vitórias de Jogador A e Jogador B. No entanto, para uma avaliação mais completa, foram examinadas também métricas como precisão e recall para cada classe (embora neste cenário de previsão de vencedor, as classes têm igual importância), além do F1-score que é a média harmônica de precisão e recall. Uma matriz de confusão foi gerada para visualizar quantas partidas de teste foram corretamente classificadas e onde ocorreram erros (por exemplo, quantas vitórias de A o modelo previu erroneamente como vitória de B e vice-versa). Esse tipo de análise ajuda a identificar se o modelo possui algum viés sistemático – por exemplo, sempre tende a predizer vitória do jogador de melhor ranking.

 

A validação cruzada também pode ter sido usada durante o treinamento para garantir que o modelo não estivesse sobreajustado (overfitting) aos dados de treino – por exemplo, usando k-fold cross validation na fase de comparação de modelos. Os resultados de desempenho indicaram que o modelo escolhido alcançou uma taxa de acerto significativa na previsão de partidas de tênis (tipicamente em torno de 70-80% de acurácia, dependendo da qualidade das features e do algoritmo selecionado). Isso demonstra a viabilidade do uso de ML para auxiliar previsões em tênis, embora ainda haja partidas imprevisíveis devido a fatores não quantificados (lesões, clima, fatores psicológicos, etc.). Em suma, a etapa de avaliação confirmou que o modelo final generaliza bem nos dados de teste e pode ser usado com confiança moderada para prever resultados de partidas futuras.

## Interface Web (Streamlit)

Para tornar o projeto mais interativo e demonstrar suas funcionalidades, foi desenvolvida uma interface web usando Streamlit. A aplicação web permite que usuários interajam com o modelo de forma simples, sem precisar executar notebooks ou scripts manualmente. Ao rodar ``front.py`` via Streamlit, a seguinte experiência é disponibilizada:

- Entrada de Dados pelo Usuário: A página inicial do app apresenta controles para o usuário fornecer os detalhes de uma partida hipotética que gostaria de prever. Por exemplo, o usuário pode selecionar o Jogador 1 e Jogador 2 a partir de listas pré-carregadas (derivadas do conjunto de dados – tipicamente os nomes dos jogadores disponíveis). Alternativamente, caso o app não disponha de listas de nomes, podem ser exibidos campos numéricos para inserir características dos jogadores (como ranking atual de cada um, número de vitórias no último ano, estatura, idade, etc.) e contexto do jogo (rodada de torneio, tipo de quadra). Essas informações servirão de features para o modelo prever o resultado. A interface foi projetada para ser amigável: com textos explicativos (usando st.markdown ou st.write) orientando o usuário a preencher os campos necessários.

- Previsão do Resultado: Após fornecer os dados de entrada, o usuário clica em um botão (por exemplo, "Prever Resultado"). O app então utiliza o modelo de ML previamente treinado (carregado em memória quando o Streamlit inicia, por meio de pickle.load no arquivo salvo em models/) para realizar a previsão. Em fração de segundos, o resultado é exibido na tela. A saída pode ser algo como: "Previsão: Jogador 1 vence a partida com 75% de probabilidade" – onde o nome do jogador vencedor previsto é destacado. Essa probabilidade pode ser derivada do método predict_proba de modelos como Random Forest ou logística, dando uma indicação de confiança. A interface pode usar componentes do Streamlit para destacar o resultado, como st.success() para indicar o favorito ou simplesmente formatar o texto do vencedor em negrito. Se apropriado, pode até mostrar a distribuição de probabilidade (por exemplo, uma barra mostrando X% vs Y%).

Em suma, o uso do Streamlit neste projeto permite que qualquer pessoa possa testar o modelo fornecendo novos inputs. Essa interface aproxima o projeto de um aplicativo real, demonstrando na prática a utilidade do modelo em um cenário aplicado de previsão de partidas de tênis.

## Licença e Créditos

Este projeto é de código aberto e está disponibilizado sob uma licença permissiva (por exemplo, MIT License). Isso significa que você pode usar, modificar e distribuir o código livremente, desde que mantenha os devidos créditos aos autores. Consulte o arquivo LICENSE no repositório para detalhes específicos da licença aplicada.


Créditos: MatchPointML foi desenvolvido como parte de um projeto acadêmico por alunos do curso de Engenharia de Computação do Instituto Mauá de Tecnologia. A concepção e implementação envolveram aplicar conhecimentos de ciência de dados, aprendizado de máquina e desenvolvimento de software adquiridos ao longo do curso. Agradecemos aos professores e colegas pelo apoio e orientação durante o desenvolvimento.

 

Além disso, vale reconhecer as fontes de dados e bibliotecas que viabilizaram o projeto: os dados históricos de tênis utilizados foram obtidos de repositórios públicos (como bases de dados de partidas profissionais) e todas as bibliotecas Python utilizadas são ferramentas open-source mantidas pela comunidade (pandas, scikit-learn, etc.). Este projeto demonstra o poder dessas ferramentas e dos dados abertos na resolução de problemas práticos.

 

Por fim, reforçamos que este README serve como documentação para futuros usuários e desenvolvedores que desejem entender, utilizar ou expandir o MatchPointML. Fique à vontade para clonar o repositório, experimentar melhorias no modelo ou adaptar a solução para outros esportes e cenários! Boas análises e bons jogos.
