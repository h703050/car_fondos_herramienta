# vamos a crear el dataframe que necesitamos de una vez con una funcion a partir del codigo isin
# por defecto los datos los buscaremos desde 1 de enero de 2000, parece suficiente

def evolucion_fondo(isin):
    import pandas as pd
    import investpy
    from datetime import date
    isin = isin
    nombre = investpy.search_funds(by='isin', value= isin).loc[0,'name']
    pais = investpy.search_funds(by='isin', value= isin).loc[0,'country']
    df = investpy.get_fund_historical_data(fund= nombre, from_date="01/01/2000",
                                       to_date=date.today().strftime("%d/%m/%Y"), country=pais)
    df = df.drop(['Open', 'High', 'Low'], axis=1)
    df['returns'] = df['Close'].pct_change()
    df = df.dropna()
    return df
    





#cogemos un data set, creamos dos columnas una con el precio Close y otra con los returns y lo denominamos df

def analisis_fondo(df):
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    from scipy.stats import norm
    import numpy as np

    #elimitamos la primera linea que contiene un NaN en los returns
    df = df.dropna()

    # calculamos tasa crecimiento anual compuesto, es el resultado de comprar y mantener
    (mu, sigma) = stats.norm.fit(df['returns'])
    Años = df['returns'].count() / 252
    # tb podriamos decir df['returns'].shape[0]
    CAGR = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1/Años) -1
    
    # Calculamos el máximo Drawdown
    Maximo_Anterior = df['Close'].cummax()
    drawdowns = 100*((df['Close'] - Maximo_Anterior ) / Maximo_Anterior)
    DD = pd.DataFrame({'Close': df['Close'],
                    'Previous Peak': Maximo_Anterior,
                   'Drawdown': drawdowns})
    
 
    resultados = (print(50*'='),
    print('> Tasa de crecimiento Anual Compuesto:', '%.6s' % (100 * CAGR),'%'),
    print('> Buy and Hold:', '%.6s' % (100*((df['Close'].iloc[-1] -
                                        df['Close'].iloc[0]) /
                                        df['Close'].iloc[0])), '%'),
    print('> Máximo Drawdown Histórico:', '%.6s' % np.min(DD['Drawdown']), '%'), 
    print('> Media Diaria:', '%.6s' % (100 * df['returns'].mean()), '%'),
    print('> Desviación Típica Diaria:', '%.6s' % (100 * df['returns'].std(ddof=1)), '%'),
    print('> Máxima Pérdida Diaria:', '%.6s' % (100 * df['returns'].min()), '%'),
    print('> Máximo beneficio Diario:', '%.6s' % (100 * df['returns'].max()), '%'),
    print('> Días Analizados:', '%.6s' % (df['returns'].shape[0])),
    print(50*'='),
    print('> Coeficiente de asimetría:', '%.6s' % df['returns'].skew()),
    print('> Curtosis:', '%.6s' % df['returns'].kurt()),
    print(50*'='),
    print('> VaR Modelo Gaussiano NC-95% :', '%.6s' % (100 * norm.ppf(0.05, mu, sigma)), '%'),
    print('> VaR Modelo Gaussiano NC-99% :', '%.6s' % (100 * norm.ppf(0.01, mu, sigma)), '%'),
    print('> VaR Modelo Gaussiano NC-99.7% :', '%.6s' % (100 * norm.ppf(0.003, mu, sigma)), '%'),
    print('> VaR Modelo Histórico NC-95% :', '%.6s' % (100 * np.percentile(df['returns'], 5)), '%'),
    print('> VaR Modelo Histórico NC-99% :', '%.6s' % (100 * np.percentile(df['returns'], 1)), '%'),
    print('> VaR Modelo Histórico NC-99.7% :', '%.6s' % (100 * np.percentile(df['returns'], 0.3)), '%'),
    print(50*'='))
    

    
    
    
def grafico_volatilidad(df):
    import matplotlib.pyplot as plt

    #Volatilidad historica de 14 dias, la misma volatilidad anualizada y la media aritmetica de la 
    # volatilidad anualizada calculada sobre los 126 valores precedentes
    df['Volatilidad_Historica_14_dias'] = 100*df['returns'].rolling(14).std()
    df['Volatilidad_14_dias_Anualizada'] = df['Volatilidad_Historica_14_dias']*(252**0.5)
    df['SMA_126_Volatilidad_Anualizada'] = df['Volatilidad_14_dias_Anualizada'].rolling(126).mean()

    # Creamos una figura con eje x e y
    fig, ax1 = plt.subplots(figsize=(15,8))
    # creamos un eje y apuesto con otra escala para una de las graficas y que se vean todas proporcionales
    ax2 = ax1.twinx()

    #creamos los objetos que vamos a dibujar en los ejes ax1 y ax2
    volatilityLine = ax1.plot(df['Volatilidad_14_dias_Anualizada'], 
                          'orange', linestyle='--', label='Vol. 14 dias Anualizada')
    smaLine = ax1.plot(df['SMA_126_Volatilidad_Anualizada'], 'green', linestyle='-', 
                  label= 'SMA 126 Vol. Anualizada')
    adjustedCloseLine = ax2.plot(df['Close'], 'black', label='Precio de cierre ajustado')

    #Titulo del grafico
    plt.title('Evolucion historica del precio y la voltilidad', fontsize=16)

    #Etiquetas de los ejes x e y
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Volatilidad Anualizada', color='black')
    ax2.set_ylabel('Precio de Cierre', color='black')

    #creamos una lista con los diferentes plots para crear un cuadro con los labels
    plotLines = volatilityLine + smaLine + adjustedCloseLine
    # Creamos una njueva lista extrajendo las etiquietas de cada plot
    labels = [line.get_label() for line in plotLines]
    # En la leyenda incluimos la lista de plots y las etiquetas de cada plot
    ax1.legend(plotLines, labels, loc='upper left', frameon=True, borderpad=1)

    ax1.grid(True)
    ax2.grid(False)

    return plt.show()


def simetria(df):
    import seaborn as sns
    from scipy import stats
    from scipy.stats import norm
    import numpy as np
    import matplotlib.pyplot as plt

    #elimitamos la primera linea que contiene un NaN en los returns
    df = df.dropna()

    #dibujamos el histograma de frecuencias
    plt.figure(figsize=(15,8))
    sns.set(color_codes = True)
    ax = sns.distplot(df['returns'], bins=100, kde=False,
    fit=stats.norm,color='green')
    #obtenemos los parametros ajustados a la distrib normal utilizados por sns
    (mu, sigma) = stats.norm.fit(df['returns'])

    # configuramos el grafico
    return plt.title('Distribucion historica de returns diarios', fontsize=16),plt.ylabel('Frecuencia'),plt.legend(['Distribucion normal. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})'.format(mu,sigma), 'Distribucion  returns'])