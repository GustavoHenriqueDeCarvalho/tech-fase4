import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import urllib.request
import requests
import seaborn as sns
from PIL import Image
from io import BytesIO


st.title('Variação do Preço por Barril do Petróleo Bruto Brent (FOB)')

multi = '''  
        Para executar o código localmente, você pode seguir os passos abaixo:

        0. Clone o repositório:  
        > git clone https://github.com/GustavoHenriqueDeCarvalho/tech-fase4.git
        1. Crie um ambiente virtual python:  
        > python -m venv myenv
        2. Ative o ambiente virtual:  
        > myenv\Scripts\activate
        3. Instale as bibliotecas necessárias:  
        > pip install -r requirements.txt
        4. Execute o código:  
        > streamlit run streamlit_app.py
'''
st.sidebar.markdown(multi)

# st.sidebar.markdown("*Streamlit* is **really** ***cool***.")
# st.sidebar.markdown('''
#     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
#     :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
# st.sidebar.markdown("Here's a bouquet &mdash;\
#             :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")


tab1, tab2, tab3, tab4, tab5 = st.tabs(['Introdução', 'Método', 'Análise do Cenário', 'Machine Learning', 'Conclusão'])

with tab1:
        multi = '''
        ## Introdução
        Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo Brent,
        que pode ser encontrado no site do Ipea. Essa base de dados histórica envolve duas colunas: data e preço (em dólares).    
        
        Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo para gerar insights
         relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.  
        Este relatório tem como objetivo analisar o comportamento do preço do petróleo brent, a fim de gerar insights para tomadas de decisões
         baseadas em dados e fornecer indicadores para um fácil acompanhamento. O preço do petróleo Brent, que é uma referência global para
          valor do petróleo, é determinado por uma combinação de fatores econômicos, políticos e ambientais. O Brent é extraído principalmente do Mar do
           Norte e serve como um benchmark para os contratos de petróleo negociados em mercados internacionais.  
        
        ## Contextualização
        O preço do Brent é influenciado por eventos como conflitos geopolíticos, decisões da Organização dos Países Exportadores de Petróleo (OPEP),
         mudanças na oferta e demanda global, flutuações cambiais e o crescimento ou desaceleração econômica mundial. Outros fatores,
          como inovações tecnológicas na extração de petróleo e as políticas de transição para energias renováveis, também afetam o mercado do petróleo Brent.   
        
        Durante crises globais, como a pandemia de COVID-19 ou tensões políticas em países produtores, o preço do Brent pode experimentar
                volatilidade significativa. Por outro lado, períodos de estabilidade política e crescimento econômico geralmente resultam em preços mais equilibrados.  
        As informações e análises apresentados dentro deste relatório apresentam dados fornecidos pelo site do ipeadata [Site - ipeadata](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) utilizamos informações do passado para entender os comportamentos e realizar previsão.'
        
        O Dashboard foi realizado utilizando o Power Bi da Microsoft onde foram realizados insights utilizando dados do inicio de 2000 até 2017. Para que seja compreendido utilizamos dados sobre demanda de energia, mortes por conflito armado e produção de petróleo.

        Clique [aqui](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) para acessar os dados do IPEA
        Clique [aqui](https://app.powerbi.com/view?r=eyJrIjoiNTg0ZDMyY2MtMzMwNi00ZDQ3LWEzY2EtMDVmZjYzZWZiYmQwIiwidCI6IjFjZTUxYjk4LWY4MmYtNGYxNy1iNDRmLTZlNzc0MDE5ZDBlOSIsImMiOjR9) para acessar o PowerBI
        Clique [aqui](https://colab.research.google.com/drive/1Gb3Ch5yoz9dnIax8BqqFZWWMX_2n6poR#scrollTo=X1RRCse9wRZI) para acessar o Machine Learning.
        '''
        
        st.markdown(multi)
with tab2:
        st.subheader('Método')
        st.write('Para a análise, utilizamos dados fornecidos pelo site do ipeadata, e através de ferramentas como o Python, foi montado um modelo de Machine Learning que realiza a previsão do preço do petróleo diariamente.')
        st.write('Ao utilizarmos destes modelos obtemos diversas vantagens como:')

        st.write('**Capacidade de Processar Grandes Volumes de Dados:** O Machine Learning pode analisar grandes quantidades de dados de maneira muito mais eficiente do que os métodos manuais ou tradicionais. Isso é essencial em um mundo cada vez mais orientado por dados, onde a quantidade e a complexidade das informações disponíveis aumentam constantemente.')
        st.write('**Capacidade de Aprendizado Automático:** Uma das principais vantagens do ML é que ele pode aprender automaticamente a partir de dados históricos sem a necessidade de programações manuais detalhadas. Isso permite que o modelo se adapte e melhore com o tempo, à medida que mais dados são disponibilizados.')
        st.write('**Automação e Eficiência:** O uso de ML para previsões reduz a necessidade de intervenção manual constante. Isso resulta em uma maior eficiência operacional, pois os modelos podem fazer previsões de forma autônoma, liberando os analistas para tarefas mais estratégicas.')
        st.write('**Análise em Tempo Real:** O ML é capaz de realizar previsões em tempo real ou quase em tempo real, o que é particularmente útil em áreas como o comércio eletrônico, a previsão do tempo, ou a análise de risco financeiro, onde decisões rápidas podem ser necessárias.')

        st.write('Além essa ferramenta, para este trabalho também estamos fazendo o uso do Streamlit uma ferramenta para criar interfaces web interativas de forma rápida e simples, especialmente voltada para aplicações de machine learning e análise de dados, casando muito bem com os nossos objetivos neste contexto.')
        st.write('Ao fazer uso desta ferramenta, também nos apropriamos das vantagens que a mesma traz, como:')

        paragraphs = [
        '**Facilidade de uso:** O Streamlit é projetado para ser extremamente simples de usar. Ele permite criar aplicações web com apenas algumas linhas de código Python.',
        '**Integração com bibliotecas Python:** O Streamlit se integra de forma muito eficaz com bibliotecas populares de Python, como pandas, matplotlib, seaborn, plotly, scikit-learn, tensorflow, entre outras. Isso torna o desenvolvimento de aplicações de análise de dados e modelos de machine learning muito mais fluido.',
        '**Desempenho e escalabilidade:** O Streamlit é projetado para ser rápido e eficiente. Ele é ótimo para protótipos e até para pequenas aplicações em produção, com boa performance mesmo em tempo real, dependendo da complexidade do modelo ou da aplicação.'
    
            ]

        for paragraph in paragraphs:
                st.write(paragraph)

        st.write('Para a demonstração visual, foi feito um Dashboard utilizando o Power Bi da Microsoft onde foram realizados insights utilizando dados de 1990 até 2017 para assim contemplar de forma igualitária todos os dados que obtivemos no decorrer deste relatório. Para que seja compreendido utilizamos dados sobre demanda de energia, mortes por conflito armado e produção de petróleo. Como dito anteriormente O Brent é extraído principalmente do Mar do Norte, por isso em nosso dashboard, filtramos o continente europeu para termos diversas visões e tirarmos os insights que possam ser usados para decisões futuras.')	

with tab3:
        # create subheader
        st.subheader('**Demanda de Energia**')
        st.write('Nesse primeiro momento de exploração alteramos a medida de energia no DataSet em relação ao apresentado, pois o valor original de MWh onde havia países com valores superiores a trilhão, onde fizemos a transformação para GWh para um melhor entendimento. Gostaríamos de demonstrar  alguns pontos de demanda de energia do Paises em relação com sua população,  observamos as seguintes informações: ')
        st.write('Se compararmos pela média da População mundial levando em conta os dados de 1950 até 2023 conseguimos observar que se não levarmos em conta a China que seria um dos países com maior quantidade de habitantes os países mais desenvolvidos ocupam a maior parte do topo do ranking de países com maiores demandas por energia. Isso nos mostra que densidade populacional não significa necessariamente mais custo com energia, um país que nos mostra isso seria o Brasil, que mesmo sendo um dos países com a maior média de população não chega a ser um dos dez maiores países com demanda de energia.')
        st.image("imagens/Demanda_energia_país01.JPG")
        st.write('Quando levantamos a produção do Óleo bruto conseguimos observar que os Estados Unidos tem uma alta demanda de energia e é um dos que mais o produz, com essa informação podemos supor com muita segurança que os Estados Unidos é um dos países que mais utilizam Petróleo para satisfazer sua demanda por energia. E ao pesquisarmos mais a fundo sobre o assunto conseguimos encontrar artigos do Governo brasileiro que confirmam que ele não é só um dos maiores consumidores como foi o maior consumidor nos anos de 2021 e 2022 ocupando a primeira posição.')
        st.subheader('Conflito Armado:')
        st.write('Infelizmente muitas são as causas que podem influenciar diretamente e indiretamente na produção e comercialização do óleo bruto, ao verificarmos o valor médio por ano conseguimos constatar que os anos que tiveram os maiores valores seriam entre os anos de 2011 a 2013, onde conseguimos identificar alguns dos conflitos que podem ter influenciado.')
        st.image("imagens/Conflito_Armado02.jpeg")
        st.write('Dentro desses anos gostaríamos de citar dois conflitos:')
        st.write('O conflito do Iraque (2011-2013) se instaurou logo após as tropas dos Estados Unidos se retirarem do território Iraquiano depois de 8 anos de guerra começaram várias revoltas da população local onde se desprendeu uma Guerra Civil que seguiu até meados de 2017. Além de ser o país com a maior quantidade de conflitos armados dentro do período citado, podemos observar a  grafico 3 e verificar que se trata de um dos 10 países que mais produzem petróleo no mundo.')
        st.write('A Guerra Civil na Líbia(2011) engloba a 16ª região que mais produzem petróleo conforme a Imagem 3, houve uma Guerra Civil entre as forças do governo regente de Muammar Gaddafi contra grupos revolucionistas populares que durou até meados do final do ano. Sendo a 9ª região com a maior média de mortos no período, de acordo com a Imagem 5 podemos citar como um dos possíveis motivos.')
        st.write('Houve outros conflitos na época como a Guerra Civil na Síria(2011) e o Conflito no Bahrein (2011-2014).')
        st.write('Poderemos ver a seguir no tópico de variação de preços como esse período de instabilidade afetou o mercado, e além disso, focando em uma visão do continente Europeu, temos os períodos com maiores números de mortes:')
        st.image("imagens/Conflito_Armado03.JPG")
        st.write('O ano de 1991 foi o mais sangrento em termos de mortes por conflito no continente Europeu, seguido por 1992, 2011 e 1990.')        
        st.write('**População e Demanda por Energia por ano**')
        st.write('Visando entender o comportamento da demanda por energia alinhado ao crescimento da população, geramos este gráfico abaixo, para ilustrar como essas informações podem se relacionar entre si e trazer importantes pontos e insights. ')
        st.image("imagens/Populacao_demanda_04.JPG")
        st.write('Temos dados consolidados para ambas as informações a partir do ano de 1990 até 2017, como podemos notar no gráfico, o crescimento da população entre 1990 e 2000 possui um crescimento linear, já a partir de 2001 o crescimento é de forma mais acelerada, e mantém esse mesmo ritmo estável até o último ano em que temos esses dados consolidados em 2017.')
        st.write('Ao observarmos a evolução de Demanda por energia podemos observar um crescimento desacelerado e estável entre os anos 1990 a 1999, já no ano 2000 ocorre um salto nesta de demanda e o crescimento desde então se mostra mais acelerado e menos estável.')
        st.write('Podemos notar que conforme a população vem crescendo, a demanda por energia também cresce, mas além disso, ocorrem outros fatores que podem causar este crescimento mais acelerado, por isso trazemos algumas hipóteses como: ')         
        st.write('**Crescimento econômico:** A década de 2000 foi marcada por um crescimento econômico robusto, especialmente em países em desenvolvimento como China e Índia, que passaram a ter um papel crescente na economia global. O aumento da produção industrial, a urbanização e o consumo elevado de bens de consumo aumentaram a necessidade de energia.')
        st.write('**Aumento do uso de eletrônicos e tecnologias:** A rápida evolução tecnológica e a proliferação de dispositivos eletrônicos principalmente no final dos anos 2000, ocasionou um grande aumento no consumo de energia relacionado ao uso de tecnologias digitais e da internet, tanto para armazenar dados quanto para a operação de servidores e centros de dados.')
        st.write('**Urbanização:** As cidades se expandiram e houve uma maior construção de infraestrutura, o que exigiu mais eletricidade para iluminação, aquecimento, refrigeração, etc.')
        st.write('Esses fatores, entre outros, resultaram em um aumento considerável da demanda por energia durante os anos 2000. Ainda que muitos países tenham adotado medidas para mitigar esse crescimento, como a implementação de fontes de energia renováveis mais tarde na década, (o que pode ter causado as mudanças conforme vemos no gráfico a partir de 2008) o aumento de consumo foi uma característica marcante desse período.')
        st.write('**Variação de preço do Petróleo**')
        st.write('Neste gráfico abaixo podemos notar as mudanças entre os menores preços praticados do petróleo, comparado aos maiores preços, desde o ano de 1990 a 2017. A partir dessa visualização, é possível perceber de forma clara grandes variações nos preços, para facilitar ainda mais o entendimento das pessoas que acessarem, ao passar o mouse pelos anos, aparecerá uma breve explicação de fatores que contribuíram para grandes mudanças ano ano.')
        st.image("imagens/Variacao_preco_petroleo05.JPG")
        st.write('A fim de trazer alguns pequenos insights, examinando os dados expostos, temos momentos de estabilidade e pequenas variações de preços, como por exemplo no ano de 1994, onde a oferta e demanda estavam equilibrados, ocasionando uma estabilidade nos preços, com o passar dos anos, é notório o aumento dessa variação, atingindo o seu pico histórico e logo em seguida uma queda acentuada no ano de 2008, tendo por explicação a crise financeira global. Outro bom exemplo é o ano de 2012, onde os preços estavam próximos, porém ambos vivenciando uma alta, devido às sanções ao Irã e instabilidade no Oriente Médio. Lembrando novamente que o dashboard traz pequenas explicações e hipóteses para cada ano descrito.')
        st.write('Outro ponto interessante para explorar são o maior e menor preço atingido no decorrer de todo o tempo aqui analisado, conforme segue abaixo na imagem:')
        st.image("imagens/Variacao_preco_petroleo06.JPG")
        st.write('Podemos reparar que o preço médio no decorrer do tempo fica em US$ 52,03.')     
with tab4:
        # -*- coding: utf-8 -*-
        """FIAP_PENULTIMO_TRABALHO.ipynb
 
        Arquivo original em:
            https://colab.research.google.com/drive/1Gb3Ch5yoz9dnIax8BqqFZWWMX_2n6poR
        """

        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # import numpy as np

        """
        
        ## Criação da tabela para usar no power bi

        ### Importando dados de mortes em guerra (conflitos armados)

        fontes:
        - vimos o dado primeiro neste site https://ourworldindata.org/grapher/annual-number-of-deaths-by-cause
        - E o site pega os dados deste outro site https://vizhub.healthdata.org/gbd-results/

        Fizemos os seguinte filtro para baixar os dados:
        """

        ConflitoArmado = pd.read_csv('content/IHME-GBD_2021_DATA-dc596585-1.csv')

        ConflitoArmado.drop(columns=['measure_id', 'measure_name', 'location_id', 'sex_id', 'sex_name', 'age_id', 'age_name', 'cause_id', 'cause_name', 'metric_id', 'metric_name', 'upper', 'lower'], inplace=True)

        #Renomeando coluna de valor para ficar mais compreensível
        ConflitoArmado.rename(columns={'val': 'MortesPorConflitoArmado', 'location_name': 'NomePais', 'year':'Ano'}, inplace=True)

        #Deixando no tipo inteiro (pois não existe 1.4 mortes, mas sim 1 morte)
        ConflitoArmado['MortesPorConflitoArmado'] = ConflitoArmado['MortesPorConflitoArmado'].astype(int)

        #Agrupando por ano e somando
        ConflitoArmado = ConflitoArmado.groupby(['NomePais', 'Ano']).sum().reset_index()

        ConflitoArmado.head()

        """##Importando dados de demanda por energia per capita"""

        DemandaPorEnergia = pd.read_csv('content/per-capita-electricity-demand.csv')

        DemandaPorEnergia = DemandaPorEnergia[['Code', 'Entity', 'Year', 'Per capita electricity demand - kWh']]

        DemandaPorEnergia.columns = ['CodigoPais', 'Pais', 'Ano', 'DemandaPorEnergiaPerCapitaKWH']

        DemandaPorEnergia = DemandaPorEnergia[['CodigoPais', 'Ano', 'DemandaPorEnergiaPerCapitaKWH']]

        DemandaPorEnergia.head()

        """##Importando dados de população"""

        populacao = pd.read_csv('content/population.csv')

        populacao = populacao[['Code', 'Entity', 'Year', 'Population - Sex: all - Age: all - Variant: estimates']]

        populacao.columns = ['CodigoPais', 'Pais', 'Ano', 'Populacao']

        populacao = populacao[['CodigoPais', 'Ano', 'Populacao']]

        populacao.head()

        """##Continentes
        fonte: https://www.kaggle.com/datasets/kirshoff/continents
        """

        continent = pd.read_csv('content/continents.csv')

        continent = continent[['alpha-3', 'name', 'region', 'sub-region', 'intermediate-region']]

        continent.columns = ['CodigoPais', 'NomePais', 'Continente', 'SubContinente', 'RegiaoIntermediaria']

        continent[continent['CodigoPais']=='BRA']

        """##Preço do petróleo Brent por ano
        Fonte: https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RBRTE&f=M (filtrando annual)
        """

        brentPrice = pd.read_csv('content/Europe_Brent_Spot_Price_FOB Annual.csv', skiprows=4).reset_index()
        brentPrice.drop(columns='index', inplace=True)
        brentPrice.columns = ['Ano', 'PrecoBarrilBrentAno']
        brentPrice.head()

        """## Extração de petróleo Brent por ano
        fonte: https://app.powerbi.com/view?r=eyJrIjoiZjY2ZjUyNzktZTU1Mi00NGIyLTliY2YtNGY1YzBjMjcxZjg0IiwidCI6ImU2ODFjNTlkLTg2OGUtNDg4Ny04MGZhLWNlMzZmMWYyMWIwZiJ9&disablecdnExpiration=1725897641
        """

        brentProduction = pd.read_csv('content/Extracao Petroleo Brent - Página1.csv')
        brentProduction.columns = ['Ano', 'TotalOilRecoveredByYearSM3']
        brentProduction.head()

        brent = pd.merge(brentPrice, brentProduction, on=['Ano'], how='left')
        brent.head()

        """##Produção de crude oil including lease condensate production (por país e ano)
        ***não é produção apenas de petróleo brent, mas ele faz parte dessa categoria***

        fonte: https://www.eia.gov/international/data/world/petroleum-and-other-liquids/more-petroleum-and-other-liquids-data?pd=5&p=00000000000000000000000000000000002&u=0&f=A&v=mapbubble&a=-&i=none&vo=value&&t=C&g=00000000000000000000000000000000000000000000000001&l=249-ruvvvvvfvtvnvv1vrvvvvfvvvvvvfvvvou20evvvvvvvvvvnvvvs0008&s=315532800000&e=1483228800000
        """

        production = pd.read_csv('content/crudeoilproductionbycountryandyear.csv', skiprows=1)
        production = production.rename(columns={'Unnamed: 1': 'NomePais'})
        production = production.iloc[2:]
        production.drop(columns='API', inplace=True)
        #pivotando as colunas de ano para apenas uma coluna
        production = production.melt(id_vars=['NomePais'], var_name='Ano', value_name='ProducaoBarrilAno')
        production['Ano'] = production['Ano'].astype('int64')
        production['ProducaoBarrilAno'] = production['ProducaoBarrilAno'].replace('--', 0).astype('float64')
        production.columns = ['NomePais', 'Ano', 'ProducaoBarrilAno_por_mb_d']
        production['NomePais'] = production['NomePais'].str.strip()
        production

        """##JOIN"""

        join = pd.merge(continent, populacao, on=['CodigoPais'], how='inner')
        join['Ano'] = join['Ano'].astype(int)
        join['Populacao'] = join['Populacao'].astype(int)

        join = pd.merge(join, DemandaPorEnergia, on=['CodigoPais', 'Ano'], how='left')

        join = pd.merge(join, ConflitoArmado, on=['NomePais', 'Ano'], how='left')

        join = pd.merge(join, production, on=['NomePais', 'Ano'], how='left')

        join

        """##Tabela final PBI"""

        join.head()

        production.head()

        PBIDash = pd.merge(join, brent, on=['Ano'], how='left')
        #PBIDash['DemandaPorEnergia_killowatshora'] = PBIDash['DemandaPorEnergiaPerCapitaKWH'] * PBIDash['Populacao']
        PBIDash['ProducaoBarrilCrudeOil_por_mil_barris_por_dia'] = PBIDash['ProducaoBarrilAno_por_mb_d']
        PBIDash['VolumeExtraidoBrent_fieldBrent_por_metros_cubicos'] = PBIDash['TotalOilRecoveredByYearSM3']
        #PBIDash.drop(columns=['DemandaPorEnergiaPerCapitaKWH', 'ProducaoBarrilAno_por_mb_d', 'TotalOilRecoveredByYearSM3'], inplace=True)
        PBIDash.drop(columns=['ProducaoBarrilAno_por_mb_d', 'TotalOilRecoveredByYearSM3'], inplace=True)
        PBIDash.head()

        PBIDash.to_csv('PBIDash.csv', index=False)

        """#Forecasting"""

        df = pd.read_csv('content/PrecoPetroleoBrentUSDPorDIA.csv', sep=';')
        df['Data'] = df['Data'].astype('datetime64[ns]')
        df['PrecoUSDBarrilBrent'] = df['PrecoUSDBarrilBrent'].str.replace(',', '.').astype('float64')
        df.info()
        df.head()

        #indica o tamanho do gráfico
        fig, ax = plt.subplots(figsize=(17, 4))
        #Coloca as linhas de grade
        ax.grid(True, color='lightgray', axis='y')
        #Cria o gráfico de linhas
        sns.lineplot(data=df, x='Data', y='PrecoUSDBarrilBrent', color='#3468FC', linewidth=2, marker='o', ax=ax)
        #Título do eixo y
        ax.set_ylabel('Preço (USD) do Barril')
        #Título do eixo x
        ax.set_xlabel('Dia')
        #título do gráfico
        ax.set_title('Preço (USD) diário do Barril de petróleo Brent')
        #mostrando o gráfico
        plt.show()

        #Como o período é muito longo, vamos filtrar o dataframe
        df_filtrado = df[df['Data']>= '2023-01-01']

        #indica o tamanho do gráfico
        fig, ax = plt.subplots(figsize=(17, 4))
        #Coloca as linhas de grade
        ax.grid(True, color='lightgray', axis='y')
        #Cria o gráfico de linhas
        sns.lineplot(data=df_filtrado, x='Data', y='PrecoUSDBarrilBrent', color='#3468FC', linewidth=2, marker='o', ax=ax)
        #Título do eixo y
        ax.set_ylabel('Preço (USD) do Barril')
        #Título do eixo x
        ax.set_xlabel('Dia')
        #título do gráfico
        ax.set_title('Preço (USD) diário do Barril de petróleo Brent')
        #mostrando o gráfico
        plt.show()

        #Vamos analisar a distribuição desses dados com o violinplot, mesclado com o boxplot
        fig, ax = plt.subplots(figsize=(20,5)) #indica o tamanho do gráfico
        ax.grid(True, color='lightgray') #Coloca as linhas de grade
        sns.boxplot(data=df_filtrado, x='PrecoUSDBarrilBrent', color='gray', boxprops=dict(alpha=.3)) #criando o boxplot
        sns.violinplot(data=df_filtrado, x='PrecoUSDBarrilBrent', color='lightgray') #criando o violinplot
        ax.set_title("Distribuição dos valores (USD) de barril de petróleo Brent") #Título do gráfico
        plt.show() #mostrando o gráfico

        """Como podemos ver, nossos dados não são muito distantes entre si.
        Vamos fazer uma decomposição desses dados. Para que possamos analisar:
        - Tendencia
        - Sazonalidade
        - Ruído
        """

        from statsmodels.tsa.seasonal import seasonal_decompose

        df_filtrado = df[df['Data']>= '2023-01-01']

        #Transformando a data em index, para facilitar a análise
        df_filtrado = df_filtrado.sort_values(by='Data')
        df_filtrado.set_index('Data', inplace=True)
        df_filtrado.head()

        resultados = seasonal_decompose(df_filtrado, model='multiplicative', period=7) #escolhida decomposição de 7 períodos pois foram uma semana inteira

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10)) #cria as 4 figuras para plotarmos os gráficos
        resultados.observed.plot(ax=ax1) #serie real
        ax1.set_title('Série Real') #título do eixo 1
        resultados.trend.plot(ax=ax2) #tendencia
        ax2.set_title('Tendencia') #título do eixo 2
        resultados.seasonal.plot(ax=ax3) #sazonalidade
        ax3.set_title('Sazonalidade') #título do eixo 3
        resultados.resid.plot(ax=ax4) #residuos
        ax4.set_title('Resíduos') #título do eixo 4

        plt.tight_layout() #Ajusta o padding entre os gráficos
        plt.show() #mostrando os gráficos

        """#Previsão dos valores de fechamento usando o modelo SARIMA

        Passos para fazer uma previsão com SARIMA:
        1. Visualizar a série temporal
        2. Identificar se os dados são estacionários
        3. Criar gráficos de correlação e auto-correlação
        4. Criar o modelo SARIMA

        ##1. Visualizar a série temporal

        Já fizemos isso acima, então simplesmente vamos colar o gráfico que criamos.
        """

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10)) #cria as 4 figuras para plotarmos os gráficos
        resultados.observed.plot(ax=ax1) #serie real
        ax1.set_title('Série Real') #título do eixo 1
        resultados.trend.plot(ax=ax2) #tendencia
        ax2.set_title('Tendencia') #título do eixo 2
        resultados.seasonal.plot(ax=ax3) #sazonalidade
        ax3.set_title('Sazonalidade') #título do eixo 3
        resultados.resid.plot(ax=ax4) #residuos
        ax4.set_title('Resíduos') #título do eixo 4

        plt.tight_layout() #Ajusta o padding entre os gráficos
        plt.show() #mostrando os gráficos

        """Agora precisamos validar se nossa série de dados é estacionária. Já vimos que ela ficou mais concentrada após os filtros que aplicamos, mas precisamos de estatística para termos certeza se ela é estacionária ou não.


        *Estacionária ou não estacionária? Vamos fazer o teste de ADF (Augmented Dickey-Fuller)*

        H0 - Hipótese nula (A série temporal NÃO é estacionária)

        H1 - Hipótese alternativa (A série temporal é estacionária)

        Vamos criar um gráfico da situação atual da nossa série, e pegar uma média móvel de 12 períodos.
        """

        df_filtrado

        ma = df_filtrado.rolling(12).mean() #pega uma média móvel a cada 12 períodos
        f, ax = plt.subplots(figsize=(20, 6)) #indica o tamanho do gráfico
        df_filtrado.plot(ax=ax, y='PrecoUSDBarrilBrent', label='PrecoUSDBarrilBrent') #cria o gráfico de linhas do fechamento
        ma.plot(ax=ax, y='PrecoUSDBarrilBrent', color="red", label='Média móvel') #cria o gráfico de linhas da média móvel
        plt.grid(True) #coloca grade no gráfico
        plt.legend() #mostra a legenda
        plt.title('Preço (USD) do Barril de petróleo Brent VS média móvel de 12 períodos') #adicionando um título ao gráfico
        plt.show() #mostrando os gráficos

        """Vamos verificar a estacionariedade da série comprovando estatísticamente usando o ADF abaixo:"""

        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.tsa.arima.model import ARIMA
        import statsmodels.api as sm
        from sklearn.metrics import mean_squared_error, r2_score

        X = df_filtrado.PrecoUSDBarrilBrent.values #pega apenas os números do fechamento
        result = adfuller(X) #Atribui a função de adfuller a variavel result

        print(f'ADF Statistic: {result[0]}') #printa o resultado de adf
        print(f'p-value: {result[1]}') #printa o p-value

        print(f'Valores críticos:') #printa os valores críticos
        for key, value in result[4].items():
            print(f'{key}: {value}')

        #Indica se a série é temporal ou não com base no resultado do p-value.
        if result[1] <= 0.05:
            print("A série temporal é estacionária.")
        else:
            print("A série temporal não é estacionária.")

        """Ótimo, nossa série já é estacionária! Vamos para a próxima etapa!

        ##3. Criar gráficos de correlação e auto-correlação
        """

        sm.graphics.tsa.plot_acf(df_filtrado, lags=50) #cria o gráfico de autocorrelação com 50 lags
        plt.show() #Para mostrar o gráfico de autocorrelação

        sm.graphics.tsa.plot_pacf(df_filtrado, lags=50) #cria o gráfico de autocorrelação parcial com 30 lags
        plt.show() #Para mostrar o gráfico de autocorrelação

        lag_acf = acf(df_filtrado.dropna(), nlags=50)
        lag_pacf = pacf(df_filtrado.dropna(), nlags=50)

        # Set desired figure size (width, height in inches)
        figsize = (14, 4)  # Adjust these values as needed

        # Create the figure with specified size
        plt.figure(figsize=figsize)
        plt.grid()

        # Plot ACF (on the left half of the figure)

        plt.plot(lag_acf, label='ACF', marker='o', markersize=2)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(df_filtrado)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(df_filtrado)), linestyle='--', color='gray')
        plt.title('Autocorrelation Function')

        # Plot PACF (on the right half of the figure)

        plt.plot(lag_pacf, label='PACF', marker='o', markersize=2)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(df_filtrado)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(df_filtrado)), linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')

        plt.legend()
        plt.show()

        """##4. Criar o modelo SARIMA"""

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        """Separando dados para treino (80%) e teste (20%)"""

        df_filtrado.head()

        # Separando treino e teste (80% treino, 20% teste)
        train_size = int(len(df_filtrado) * 0.8)
        train, test = df_filtrado[:train_size], df_filtrado[train_size:]

        #validando
        train.head()

        """Criando modelo"""

        # Definir os parâmetros baseados na análise
        p, d, q = 1, 0, 1  # Parâmetros não sazonais
        P, D, Q, m = 1, 0, 1, 7  # Parâmetros sazonais

        # Criar o modelo SARIMA
        model = SARIMAX(train['PrecoUSDBarrilBrent'],
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, m))

        # Ajustar o modelo
        results = model.fit(disp=False)

        # Exibir o sumário do modelo ajustado
        print(results.summary())

        forecast = results.get_forecast(steps=test.size)
        forecast_index = test.index
        forecast = forecast.predicted_mean  # Obtenha somente os valores previstos
        forecast.index = forecast_index  # Ajuste o índice de datas correto
        plt.figure(figsize=(10, 6))
        plt.plot(train['PrecoUSDBarrilBrent'], label="Treino")
        plt.plot(test['PrecoUSDBarrilBrent'], label="Teste", color="orange")
        plt.plot(forecast, label="Previsão", color="green")
        plt.legend()
        plt.title("Previsão com SARIMA")
        plt.show()

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Exemplo comparando previsões com o conjunto de validação
        mae = mean_absolute_error(test['PrecoUSDBarrilBrent'], forecast)
        rmse = mean_squared_error(test['PrecoUSDBarrilBrent'], forecast, squared=False)

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        """Como foi visto, o modelo SARIMA não foi muito bom para nosssa previsão, portanto vamos utilizar um modelo mais robusto.

        ##5. Criar o modelo LSTM
        """


        df_filtrado.head()

        # Separando treino e teste (90% treino, 10% teste)
        train_size = int(len(df_filtrado) * 0.9)
        train_data, test_data = df_filtrado[:train_size], df_filtrado[train_size:]

        print(f'Linhas Treino {len(train_data)}')
        print(f'Linhas Teste {len(test_data)}')
        train.head()

        # Importando bibliotecas necessárias
        from sklearn.preprocessing import MinMaxScaler  # Para escalonamento de dados
        import numpy as np  # Biblioteca para operações de álgebra linear

        # Inicializando o objeto MinMaxScaler
        scaler = MinMaxScaler()
        # Aplicando o escalonamento aos dados de treino.
        # Isso transforma os dados para que fiquem dentro do intervalo [0, 1], o que é útil para melhorar a performance do modelo.
        train_scaled = scaler.fit_transform(train)
        # Redimensionando os dados para um vetor unidimensional
        train_scaled_reshaped = train_scaled.reshape(-1)

        # Definindo o número de timesteps.
        # timesteps é o número de pontos de tempo anteriores usados para prever o próximo ponto.
        timesteps = 7

        # Inicializando listas para armazenar os dados de entrada (X_train) e saída (y_train) do modelo
        X_train = []
        y_train = []

        # Loop para criar sequências de entrada e saídas correspondentes
        for i in range(timesteps, train.shape[0]):
            # Para cada ponto no conjunto de dados, exceto os últimos 'timesteps' pontos,
            # cria uma sequência de 'timesteps' pontos anteriores como entrada
            X_train.append(train_scaled[i-timesteps:i, 0])  # Adicionando a sequência de entrada

            # O valor imediatamente seguinte a essa sequência é usado como saída
            y_train.append(train_scaled[i, 0])  # Adicionando o valor de saída correspondente

        # Convertendo as listas para arrays NumPy para uso em modelos de aprendizado de máquina
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Resumo: Este código está preparando os dados de entrada e saída para treinar um modelo de séries temporais.
        # Cada entrada é uma sequência de 7 pontos de dados, e a saída correspondente é o ponto de dados
        # que segue esses 7 pontos na série.
        X_train, y_train

        from numpy.random import seed

        # Reshaping X_train for Neural Network Input

        # X_train é um array NumPy contendo os dados de entrada para o modelo de rede neural.
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        # Esta linha remodela X_train para a forma (número de amostras, timesteps, número de características por timestep).
        # Em redes neurais recorrentes como LSTM, espera-se que a entrada seja um tensor 3D com essa forma.
        # Aqui, estamos configurando apenas uma característica por timestep (daí o '1' no final).

        # Definindo uma semente para reprodutibilidade
        seed(20240729)

        # Para esse modelo LSTM inicial, usaremos a biblioteca Keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        from sklearn.preprocessing import MinMaxScaler


        # O escalonamento é uma etapa importante ao trabalhar com redes neurais, pois ajuda a normalizar os dados e melhorar a convergência do modelo.
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)


        X_train = []
        y_train = []

        # Estruturação dos dados de treinamento para se adequarem ao modelo de rede neural recorrente
        for i in range(timesteps, train.shape[0]):
            X_train.append(train_scaled[i-timesteps:i, 0])  # Adicionando a sequência de timesteps anteriores
            y_train.append(train_scaled[i, 0])              # Adicionando o valor atual como rótulo

        # Convertendo as listas em arrays NumPy para uso no modelo de aprendizado de máquina
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Construção do modelo sequencial usando Keras
        model = Sequential()

        # Adicionando camadas à rede neural para previsão de séries temporais

        # Adicionando a primeira camada LSTM (Long Short-Term Memory)
        model.add(LSTM(
            units = 50,  # Número de unidades (neurônios) na camada, um indicador da "capacidade" da camada
            return_sequences = True,  # Mantém as sequências completas como saída para a próxima camada LSTM
            input_shape = (X_train.shape[1], 1)  # Define a forma da entrada (número de timesteps e recursos por timestep)
        ))

        # A camada LSTM é crucial para modelar dependências em sequências temporais, como preços.
        # Ela é capaz de aprender a partir de longas sequências de dados, mantendo informações importantes e esquecendo as irrelevantes.

        # Adicionando a camada de Dropout para regularização
        model.add(Dropout(0.25))  # Descarta 25% das unidades aleatoriamente para prevenir overfitting

        # O Dropout é uma técnica de regularização eficaz para reduzir o overfitting em redes neurais,
        # especialmente útil em redes profundas e complexas.

        # Adicionando mais camadas LSTM e Dropout
        # Essas camadas adicionais aumentam a capacidade do modelo de aprender padrões complexos nos dados.

        model.add(LSTM(units = 50, return_sequences = True))  # Outra camada LSTM
        model.add(Dropout(0.25))  # Dropout aumentado para 25% para maior regularização

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))  # Dropout de 20%

        # A última camada LSTM não tem return_sequences = True,
        # isso indica que a sequência completa não é mais necessária nas saídas
        model.add(LSTM(units = 50))  # Última camada LSTM

        # Adicionando outra camada de Dropout
        model.add(Dropout(0.25))  # Dropout de 25%

        # A estrutura de múltiplas camadas LSTM seguidas por Dropout é uma configuração comum em tarefas de previsão de séries temporais.
        # Isso permite que o modelo capture padrões complexos e de longo alcance nos dados,
        # enquanto o Dropout ajuda a garantir que o modelo não se ajuste excessivamente aos dados de treinamento.

        # Adicionando a camada de saída
        model.add(Dense(units = 1))

        # Compilando a rede neural recorrente
        # Usando 'adam' como otimizador e 'mean_squared_error' como função de perda
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Treinando o modelo com os dados de treinamento
        # A quantidade de de epochs foi baseada em: https://medium.com/@yousufdata/6-tips-to-tweak-your-lstm-bilstm-15fd02685c8
        model.fit(X_train, y_train, epochs = 120, batch_size = 32)

        test_data.head()

        # Preparação dos dados de teste
        # Extraindo os preços reais das ações (valor de fechamento) do conjunto de teste
        preco_real = test_data.values

        # Combinando os dados de treinamento e teste para formar sequências de entrada para a previsão
        combine = pd.concat((train_data['PrecoUSDBarrilBrent'], test_data['PrecoUSDBarrilBrent']), axis = 0)

        # Preparando as entradas de teste, incluindo os preços das ações dos últimos timesteps
        test_inputs = combine[len(combine) - len(test_data) - timesteps:].values
        test_inputs = test_inputs.reshape(-1,1)
        test_inputs = scaler.transform(test_inputs)

        # Processando os dados de teste da mesma forma que os dados de treinamento
        X_test = []
        for i in range(timesteps, test_data.shape[0]+timesteps):
            X_test.append(test_inputs[i-timesteps:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Realizando previsões com o modelo treinado
        preco_previsto = model.predict(X_test)
        preco_previsto = scaler.inverse_transform(preco_previsto)



        from sklearn.metrics import r2_score


        ## Calculando o R2 para comparar com o modelo ARIMA que implementamos anteriormente
        r2 = r2_score(preco_real, preco_previsto)
        print(f'R2: {r2}')

        """O modelo LSTM também não obteve resultados aceitáveis, está melhor que o SARIMA, mas o R2 está bem baixo.

        Vamos tentar outro modelo.

        ##Criar o modelo PROPHET
        """

        from prophet import Prophet
        from sklearn.metrics import mean_absolute_error

        df_filtrado.head()

        # Preparar os dados
        df_prophet = df_filtrado.copy()
        df_prophet.reset_index(inplace=True)
        df_prophet.columns = ['ds', 'y']

        # Suavizar a série (média móvel)
        df_prophet['y'] = df_prophet['y'].rolling(window=7, center=False).mean()
        df_prophet.dropna(inplace=True)

        # Configurar modelo com ajustes
        model = Prophet(
            daily_seasonality=False,
            changepoint_prior_scale=0.5,  # Mais flexível com mudanças de tendência
            uncertainty_samples=100  # Reduz amostras de incerteza
        )
        model.add_seasonality(name='weekly', period=7, fourier_order=10)  # Ajustar sazonalidade semanal

        # Ajustar o modelo
        model.fit(df_prophet)

        # Criar previsões futuras
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)

        # Visualizar os resultados
        fig = model.plot(forecast)
        plt.title("Previsões Refinadas com Prophet (15 dias)", fontsize=14)
        plt.show()

        forecast_train = forecast.loc[forecast['ds'].isin(df_prophet['ds'])]
        y_true = df_prophet['y'].values  # Valores reais
        y_pred = forecast_train['yhat'].values  # Previsões do modelo (yhat é a previsão central)

        # Ver componentes
        fig_components = model.plot_components(forecast)
        #create y_true
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Erro Absoluto Médio (MAE): {mae}")
with tab5:
        paragraphs = [
        '1º**Demanda de energia por país:** a grande demanda por energia no Reino Unido entre 1990 e 2017 foi impulsionada pelo crescimento econômico, mudanças no estilo de vida e padrões de consumo, aumento da urbanização, bem como o impacto climático e o aumento da população. A transformação da economia, com o crescimento dos setores de serviços e tecnologia, também foi um fator-chave nesse aumento da demanda energética.',
        '2º**Conflito Armado:** Os conflitos armados têm um impacto significativo no preço do petróleo Brent devido a vários fatores econômicos, geopolíticos e de oferta e demanda.    Qualquer incerteza sobre a oferta ou a segurança do fornecimento de petróleo pode gerar uma reação imediata dos mercados. A diminuição da oferta, a elevação do risco geopolítico, o impacto nos transportes e as sanções econômicas especialmente em regiões-chave produtoras contribuem para a volatilidade do preço do petróleo, levando-o a aumentar.',
        '3º**População e Demanda por Energia por ano:** Não apenas a população cresce, mas além disso, também ocorrem mudanças no estilo de vida e diversas evoluções tecnológicas, eletrônicas, industriais e urbanas, que expandem a demanda por energia ano após ano. Estes fatores ligados com a crescente população tornam a demanda cada vez maior. ',
        '4º**Variação de preço do Petróleo:** O petróleo é uma commodity global essencial e altamente sensível a qualquer instabilidade, por isso são diversos os fatores que podem causar uma variação do preço, desde crises econômicas, conflitos armados até a maiores demandas e crescimento exponencial da população com maiores inovações tecnológicas.'
            ]

        for paragraph in paragraphs:
                st.write(paragraph)
        st.write('**Referências Extras**')
        st.write('Disponível em: [https://www.eia.gov](http://https://www.eia.gov). Acesso em:20 nov. 2024.')
        st.write('Disponível em: [http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view). Acesso em:29 out. 2024.')
        st.write('Disponível em: [https://www.gov.br/anp/pt-br/centrais-de-conteudo/publicacoes/anuario-estatistico/arquivos-anuario-estatistico-2023/secao-1/secao-1.pdf](https://www.gov.br/anp/pt-br/centrais-de-conteudo/publicacoes/anuario-estatistico/arquivos-anuario-estatistico-2023/secao-1/secao-1.pdf). Acesso em:04 nov. 2024.')
        st.write('Disponível em: [https://mundoeducacao.uol.com.br/historiageral/guerra-civil-na-libia.htm](https://mundoeducacao.uol.com.br/historiageral/guerra-civil-na-libia.htm). Acesso em:19 nov. 2024.')
        st.write('Disponível em: [https://sites.ufpe.br/oci/2022/06/13/guerra-civil-na-libia-2011/](https://sites.ufpe.br/oci/2022/06/13/guerra-civil-na-libia-2011/). Acesso em:19 nov. 2024.')
        st.write('Disponível em: [https://www.bbc.com/portuguese/articles/c84m8d4xdzgo](https://www.bbc.com/portuguese/articles/c84m8d4xdzgo). Acesso em:19 nov. 2024.')
        st.write('Disponível em: [https://www.bbc.com/portuguese/internacional-55351024](https://www.bbc.com/portuguese/internacional-55351024). Acesso em:19 nov. 2024.')
