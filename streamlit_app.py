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
        st.subheader('Introdução')
        multi = '''
        Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo Brent,
        que pode ser encontrado no site do Ipea. Essa base de dados histórica envolve duas colunas: data e preço (em dólares).    
        
        Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo para gerar insights
         relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.  
        Este relatório tem como objetivo analisar o comportamento do preço do petróleo brent, a fim de gerar insights para tomadas de decisões
         baseadas em dados e fornecer indicadores para um fácil acompanhamento. O preço do petróleo Brent, que é uma referência global para
          valor do petróleo, é determinado por uma combinação de fatores econômicos, políticos e ambientais. O Brent é extraído principalmente do Mar do
           Norte e serve como um benchmark para os contratos de petróleo negociados em mercados internacionais.  
        
        O preço do Brent é influenciado por eventos como conflitos geopolíticos, decisões da Organização dos Países Exportadores de Petróleo (OPEP),
         mudanças na oferta e demanda global, flutuações cambiais e o crescimento ou desaceleração econômica mundial. Outros fatores,
          como inovações tecnológicas na extração de petróleo e as políticas de transição para energias renováveis, também afetam o mercado do petróleo Brent.   
        
        Durante crises globais, como a pandemia de COVID-19 ou tensões políticas em países produtores, o preço do Brent pode experimentar
                volatilidade significativa. Por outro lado, períodos de estabilidade política e crescimento econômico geralmente resultam em preços mais equilibrados.  
        As informações e análises apresentados dentro deste relatório apresentam dados fornecidos pelo site do ipeadata [Site - ipeadata](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) utilizamos informações do passado para entender os comportamentos e realizar previsão.'
        
        O Dashboard foi realizado utilizando o Power Bi da Microsoft onde foram realizados insights utilizando dados do inicio de 2000 até 2017. Para que seja compreendido utilizamos dados sobre demanda de energia, mortes por conflito armado e produção de petróleo.
        '''
        st.write('Clique [aqui](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) para acessar os dados do IPEA)'
        st.write('Clique [aqui](https://app.powerbi.com/view?r=eyJrIjoiNTg0ZDMyY2MtMzMwNi00ZDQ3LWEzY2EtMDVmZjYzZWZiYmQwIiwidCI6IjFjZTUxYjk4LWY4MmYtNGYxNy1iNDRmLTZlNzc0MDE5ZDBlOSIsImMiOjR9) para acessar o PowerBI)'
        st.write('Clique [aqui](https://colab.research.google.com/drive/1Gb3Ch5yoz9dnIax8BqqFZWWMX_2n6poR#scrollTo=X1RRCse9wRZI) para acessar o Machine Learning.)'
        
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
        st.subheader('Demanda de Energia')
        st.write('Nesse primeiro momento de exploração alteramos a medida de energia no DataSet em relação ao apresentado, pois o valor original de MWh onde havia países com valores superiores a trilhão, onde fizemos a transformação para GWh para um melhor entendimento. Gostaríamos de demonstrar  alguns pontos de demanda de energia do Paises em relação com sua população,  observamos as seguintes informações: ')
        st.write('Se compararmos pela média da População mundial levando em conta os dados de 1950 até 2023 conseguimos observar que se não levarmos em conta a China que seria um dos países com maior quantidade de habitantes os países mais desenvolvidos ocupam a maior parte do topo do ranking de países com maiores demandas por energia. Isso nos mostra que densidade populacional não significa necessariamente mais custo com energia, um país que nos mostra isso seria o Brasil, que mesmo sendo um dos países com a maior média de população não chega a ser um dos dez maiores países com demanda de energia.')
        #colocar graficos
        st.write('Quando levantamos a produção do Óleo bruto conseguimos observar que os Estados Unidos tem uma alta demanda de energia e é um dos que mais o produz, com essa informação podemos supor com muita segurança que os Estados Unidos é um dos países que mais utilizam Petróleo para satisfazer sua demanda por energia. E ao pesquisarmos mais a fundo sobre o assunto conseguimos encontrar artigos do Governo brasileiro que confirmam que ele não é só um dos maiores consumidores como foi o maior consumidor nos anos de 2021 e 2022 ocupando a primeira posição.')
        st.subheader('Conflito Armado:')
        #colocar graficos
        # inserir imagem
        st.write('Infelizmente muitas são as causas que podem influenciar diretamente e indiretamente na produção e comercialização do óleo bruto, ao verificarmos o valor médio por ano conseguimos constatar que os anos que tiveram os maiores valores seriam entre os anos de 2011 a 2013, onde conseguimos identificar alguns dos conflitos que podem ter influenciado.')
        #colocar graficos
        st.write('Dentro desses anos gostaríamos de citar dois conflitos:')
        st.write('O conflito do Iraque (2011-2013) se instaurou logo após as tropas dos Estados Unidos se retirarem do território Iraquiano depois de 8 anos de guerra começaram várias revoltas da população local onde se desprendeu uma Guerra Civil que seguiu até meados de 2017. Além de ser o país com a maior quantidade de conflitos armados dentro do período citado, podemos observar a  grafico 3 e verificar que se trata de um dos 10 países que mais produzem petróleo no mundo.')
        st.write('A Guerra Civil na Líbia(2011) engloba a 16ª região que mais produzem petróleo conforme a Imagem 3, houve uma Guerra Civil entre as forças do governo regente de Muammar Gaddafi contra grupos revolucionistas populares que durou até meados do final do ano. Sendo a 9ª região com a maior média de mortos no período, de acordo com a Imagem 5 podemos citar como um dos possíveis motivos.')
        st.write('Houve outros conflitos na época como a Guerra Civil na Síria(2011) e o Conflito no Bahrein (2011-2014).')
        st.write('Poderemos ver a seguir no tópico de variação de preços como esse período de instabilidade afetou o mercado, e além disso, focando em uma visão do continente Europeu, temos os períodos com maiores números de mortes:')
        #imagem
        st.write('O ano de 1991 foi o mais sangrento em termos de mortes por conflito no continente Europeu, seguido por 1992, 2011 e 1990.')        
        st.write('**População e Demanda por Energia por ano**')
        st.write('Visando entender o comportamento da demanda por energia alinhado ao crescimento da população, geramos este gráfico abaixo, para ilustrar como essas informações podem se relacionar entre si e trazer importantes pontos e insights. ')
        st.write('Temos dados consolidados para ambas as informações a partir do ano de 1990 até 2017, como podemos notar no gráfico, o crescimento da população entre 1990 e 2000 possui um crescimento linear, já a partir de 2001 o crescimento é de forma mais acelerada, e mantém esse mesmo ritmo estável até o último ano em que temos esses dados consolidados em 2017.')
        st.write('Ao observarmos a evolução de Demanda por energia podemos observar um crescimento desacelerado e estável entre os anos 1990 a 1999, já no ano 2000 ocorre um salto nesta de demanda e o crescimento desde então se mostra mais acelerado e menos estável.')
        st.write('Podemos notar que conforme a população vem crescendo, a demanda por energia também cresce, mas além disso, ocorrem outros fatores que podem causar este crescimento mais acelerado, por isso trazemos algumas hipóteses como: ')         
        st.write('**Crescimento econômico:** A década de 2000 foi marcada por um crescimento econômico robusto, especialmente em países em desenvolvimento como China e Índia, que passaram a ter um papel crescente na economia global. O aumento da produção industrial, a urbanização e o consumo elevado de bens de consumo aumentaram a necessidade de energia.')
        st.write('**Aumento do uso de eletrônicos e tecnologias:** A rápida evolução tecnológica e a proliferação de dispositivos eletrônicos principalmente no final dos anos 2000, ocasionou um grande aumento no consumo de energia relacionado ao uso de tecnologias digitais e da internet, tanto para armazenar dados quanto para a operação de servidores e centros de dados.')
        st.write('**Urbanização:** As cidades se expandiram e houve uma maior construção de infraestrutura, o que exigiu mais eletricidade para iluminação, aquecimento, refrigeração, etc.')
        st.write('Esses fatores, entre outros, resultaram em um aumento considerável da demanda por energia durante os anos 2000. Ainda que muitos países tenham adotado medidas para mitigar esse crescimento, como a implementação de fontes de energia renováveis mais tarde na década, (o que pode ter causado as mudanças conforme vemos no gráfico a partir de 2008) o aumento de consumo foi uma característica marcante desse período.')
        st.write('**Variação de preço do Petróleo**')
        st.write('Neste gráfico abaixo podemos notar as mudanças entre os menores preços praticados do petróleo, comparado aos maiores preços, desde o ano de 1990 a 2017. A partir dessa visualização, é possível perceber de forma clara grandes variações nos preços, para facilitar ainda mais o entendimento das pessoas que acessarem, ao passar o mouse pelos anos, aparecerá uma breve explicação de fatores que contribuíram para grandes mudanças ano ano.')
        #colocar graficos
        st.write('A fim de trazer alguns pequenos insights, examinando os dados expostos, temos momentos de estabilidade e pequenas variações de preços, como por exemplo no ano de 1994, onde a oferta e demanda estavam equilibrados, ocasionando uma estabilidade nos preços, com o passar dos anos, é notório o aumento dessa variação, atingindo o seu pico histórico e logo em seguida uma queda acentuada no ano de 2008, tendo por explicação a crise financeira global. Outro bom exemplo é o ano de 2012, onde os preços estavam próximos, porém ambos vivenciando uma alta, devido às sanções ao Irã e instabilidade no Oriente Médio. Lembrando novamente que o dashboard traz pequenas explicações e hipóteses para cada ano descrito.')
        st.write('Outro ponto interessante para explorar são o maior e menor preço atingido no decorrer de todo o tempo aqui analisado, conforme segue abaixo na imagem:')
        #colocar graficos
        st.write('Podemos reparar que o preço médio no decorrer do tempo fica em US$ 52,03.')     
with tab4:
        st.write('Machine Learning')
        # -*- coding: utf-8 -*-
        """FIAP_PENULTIMO_TRABALHO.ipynb

        Automatically generated by Colab.

        Original file is located at
            https://colab.research.google.com/drive/1Gb3Ch5yoz9dnIax8BqqFZWWMX_2n6poR
        """

        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # import numpy as np

        """# Criação da tabela para usar no power bi

        ##Importando dados de mortes em guerra (conflitos armados)

        fontes:
        - vimos o dado primeiro neste site https://ourworldindata.org/grapher/annual-number-of-deaths-by-cause
        - E o site pega os dados deste outro site https://vizhub.healthdata.org/gbd-results/

        Fizemos os seguinte filtro para baixar os dados:
        ![image.png](data:image/png;base64,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)
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

        # Ver componentes
        fig_components = model.plot_components(forecast)

        mae = mean_absolute_error(y_true, y_pred)
        print(f"Erro Absoluto Médio (MAE): {mae}")
with tab5:
        st.write('Conclusão')
        paragraphs = [
        '1º**Demanda de energia por país:** a grande demanda por energia no Reino Unido entre 1990 e 2017 foi impulsionada pelo crescimento econômico, mudanças no estilo de vida e padrões de consumo, aumento da urbanização, bem como o impacto climático e o aumento da população. A transformação da economia, com o crescimento dos setores de serviços e tecnologia, também foi um fator-chave nesse aumento da demanda energética.',
        '2º**Conflito Armado:**Os conflitos armados têm um impacto significativo no preço do petróleo Brent devido a vários fatores econômicos, geopolíticos e de oferta e demanda.    Qualquer incerteza sobre a oferta ou a segurança do fornecimento de petróleo pode gerar uma reação imediata dos mercados. A diminuição da oferta, a elevação do risco geopolítico, o impacto nos transportes e as sanções econômicas especialmente em regiões-chave produtoras contribuem para a volatilidade do preço do petróleo, levando-o a aumentar.',
        '3º**População e Demanda por Energia por ano:**Não apenas a população cresce, mas além disso, também ocorrem mudanças no estilo de vida e diversas evoluções tecnológicas, eletrônicas, industriais e urbanas, que expandem a demanda por energia ano após ano. Estes fatores ligados com a crescente população tornam a demanda cada vez maior. ',
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
