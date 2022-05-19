import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import datetime as dt
from matplotlib import rc

path = f'{os.getcwd()}/DATA/'
path_graph = f'{os.getcwd()}/UTILS/Images/'


# Función para importar archivo
def import_file(path, file_name):
    file = file_name
    x = pd.read_csv(path+file)

    return x


# Función para ver los valores nulos
def missing_data(x):
    '''
   Objetivo: encontrar una relación de los valores nulos de un dataset.

   args.
   ----
   data: dataset

   return.
   ----
   tabla con la relación de NaN
   '''
    total_null = x.isnull().sum().sort_values(ascending=False)
    percentage_null = (x.isnull().sum()/x.isnull().count()).sort_values(ascending=False)
    return pd.concat([total_null,percentage_null], axis=1, keys=['Total', 'Percent'])


# Definimos una función para acortar los nombres de los países
def rename_countries(x,new_column):
    '''
    Objetivo: Simplificar los nombres de alguno países de un dataset.

    Args:
    ----
    data: dataset, nombre de la columna
    '''
    country_rename = {
        'United States':'USA',
        'United Kingdom': 'UK',
        'South Korea': 'S.Korea',
        'United Arab Emirates': 'UAE',
        'Czech Republic': 'Czechia',
        'West Germany': 'Germany',
        'Soviet Union': 'Russia',
        'South Africa': 'S.Africa'
        }
    return x[new_column].replace(country_rename,inplace=True)


# Creamos una nueva columna que trae el primer valor que aparece en la columna 'country'
def clean_country(x,column,new_column):
    '''
    Objetivo: limpiar la columna de países.

    args:
    ----
    data: dataset, nombre de la columna.

    Resultado:
    ----
    Se crea una nueva columna en el dataset.
    '''
    # Separamos los distintos países en columnas y nos quedamos solo con el primer valor:
    x[new_column] = x[column].apply(lambda x: x.split(",")[0])
    # Ahora podemos eliminar la columna 'country' porque pasaremos a utilizar la columna 'country_1':
    del x[column]
    # Creamos una lista con todos los países que tengan al menos un espacio. La idea es simplificar 
    # sus nombres (por ejemplo: United States pasará a ser USA):
    countries = x[new_column].unique()
    country_space = []
    for country in countries:
        if country.__contains__(' '):
            country_space.append(country)
    rename_countries(x,new_column)

# Definimos una función para simplificar la columna 'rating'
def rename_rating(x,col_rating):
    '''
    Objetivo: Simplificar los nombres de alguno países de un dataset.

    Args:
    ----
    data: dataset, nombre de la columna
    '''
    rating_rename = {
    'TV-PG': 'Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Kids',
    'TV-Y7': 'Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Toddlers',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
    }
    return x[col_rating].replace(rating_rename, inplace=True)


def clean_categories(x,col_name,new_col_name):
    '''
    Objetivo: Simplificar los nombres de las categorías.

    Args:
    ----
    data: dataset, nombre de la columna a ser tratada, nombre de la nueva columna creada
    '''
    categories = x[col_name].str.split(",",expand=True) # separa las listas por coma y crea columnas
    

    categories[0].replace(regex=r'International', value=np.nan, inplace=True)
    categories[1].replace(regex=r'International', value=np.nan, inplace=True)
    categories[0].replace(regex=r'British', value=np.nan, inplace=True)
    categories[1].replace(regex=r'British', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Independent', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Independent', value=np.nan, inplace=True)
    categories[0].replace(regex=r'LGBTQ', value=np.nan, inplace=True)
    categories[1].replace(regex=r'LGBTQ', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Classic', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Classic', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Kid', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Kid', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Children', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Children', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Spanish', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Spanish', value=np.nan, inplace=True)
    categories[0].replace(regex=r'Korean', value=np.nan, inplace=True)
    categories[1].replace(regex=r'Korean', value=np.nan, inplace=True)
    categories[0].replace('^\s+', '', regex=True, inplace=True)
    categories[1].replace('^\s+', '', regex=True, inplace=True)
    categories[2].replace('^\s+', '', regex=True, inplace=True)
    
    # Después de definir los valores NaN, usamos la función bfill para rellenar las líneas con la información de la columna más a derecha que no sea NaN.
    categories.bfill(axis ='columns',inplace=True)

    categories_simplified = {
    'Docuseries': 'Documentaries',
    'TV Dramas': 'Drama',
    'Crime TV Shows': 'Thriller',
    'Crime TV Show': 'Thriller',
    'Crime Movies': 'Thriller',
    'Romantic TV Shows': 'Romance',
    'Dramas': 'Drama',
    'Reality TV': 'Reality Show',
    'TV Comedies': 'Comedy',
    'TV Action & Adventure': 'Action & Adventure',
    'Horror Movies': 'Horror',
    'Thrillers': 'Thriller',
    'TV Sci-Fi & Fantasy': 'Sci-Fi & Fantasy',
    'Music & Musicals': 'Musical',
    'TV Shows': 'Other',
    'Stand-Up Comedy & Talk Shows': 'Stand-Up Comedy',
    'Movies': 'Other',
    'Sports Movies': 'Sport',
    'Anime Features': 'Anime',
    'Romantic Movies': 'Romance',
    'TV Mysteries': 'Thriller',
    'Cult Movies': 'Cult',
    'TV Horror': 'Horror',
    'Faith & Spirituality': 'Spirituality',
    'TV Thrillers': 'Thriller',
    'International TV Shows': 'Other',
    "Kids' TV": 'Kids',
    'Teen TV Shows': 'Teen',
    'Spanish-Language TV Shows': 'Other',
    'Romantic Comedies': 'Romance',
    'Romantic TV Show': 'Romance',
    'Children & Family': 'Kids',
    'Children & Family Movies': 'Kids',
    'Sci-Fi Movies': 'Sci-Fi & Fantasy',
    'Sci-Fi': 'Sci-Fi & Fantasy',
    'Sci-FI & Fantasy': 'Sci-Fi & Fantasy',
    'Fantasy': 'Sci-Fi & Fantasy',
    'Anime Series': 'Anime',
    'Comedies': 'Comedy',
    None: 'Other'
    }
    categories[0].replace(categories_simplified, inplace=True)

    x[new_col_name] = categories[0]


# Creamos el primer gráfico de distribución de tipo de contenido.

def plot_distribution(x, col_name, var1, var2, name_file):
    '''
    Objetivo: Construir un gráfico con la distribución en porcentaje de la cantidad de películas y series en Netflix.

    Args:
    ----
    data: dataset, nombre de la columna utilizada, 'Movie', 'TV Show', nombre del archivo del gráfico para guardarlo
    '''
    # Preparamos los datos - porcentajes de películas x series:
    type_title = x.groupby([col_name])[col_name].count()
    ratio_title=pd.DataFrame(((type_title/len(x))).round(2)).T

    # Definimos el tamaño del gráfico:
    fig, ax = plt.subplots(1,1,figsize=(10, 3))

    # Pintamos las barras
    ax.barh(ratio_title.index, ratio_title[var1], alpha=0.8,
        color='#db0000')
    ax.barh(ratio_title.index, ratio_title[var2], left=ratio_title[var1], alpha=0.8,
    color='#564d4d')

    # movie percentage
    for i in ratio_title.index:
        ax.annotate(f"{int(ratio_title[var1][i]*100)}%", 
                    xy=(ratio_title[var1][i]/2, i),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')

        ax.annotate("Movies", 
                    xy=(ratio_title[var1][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif', color='white')
    
    
    for i in ratio_title.index:
        ax.annotate(f"{int(ratio_title[var2][i]*100)}%", 
                    xy=(ratio_title[var1][i]+ratio_title[var2][i]/2, i),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')
        ax.annotate("Show", 
                    xy=(ratio_title[var1][i]+ratio_title[var2][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')

    ax.set_xticklabels('', fontfamily='serif', rotation=0, color='white')
    ax.set_yticklabels('', fontfamily='serif', rotation=0, color='white')

    plt.savefig(path_graph + name_file+'.png',transparent=True)

    return plt.show();


# Definimos una función para crear un gráfico con la evolución en la oferta de contenidos a lo largo de los años.
def bar_graph_1(var, name_file, ytick_1=0, ytick_2=2000, ytick_3=250, rotation_=45, color_='white', color_bar='#db0000'):
    '''
    Objetivo: Construir un gráfico con evolución de los contenidos a lo largo de los años.

    Args:
    ----
    data: dataset agrupado, nombre del archivo que se guardará, ytick_1 = donde empieza el eje y, ytick_2 = donde termina el eje y, ytick_3 = frecuencia 
    '''

    plt.figure(figsize=(10,5))

    # create a dataset
    height = var
    bars = var.index
    x_pos = np.arange(len(bars))

    # Create bars
    plt.bar(x_pos, height, color=color_bar)

    # Create names on the x-axis
    plt.xticks(x_pos, bars, color=color_, rotation=rotation_)
    plt.yticks(np.arange(ytick_1,ytick_2,ytick_3), color=color_)


    plt.savefig(path_graph + name_file+'.png',transparent=True)

    return plt.show();


def movie_show_percountry(x, name_file, color_='white'):
        # Preparamos los datos:
        # Seleccionamos los 10 países con más títulos en netflix.
        country = x['prod_country'].value_counts()[:11].index
        # Ahora sacamos un nuevo dataset que nos enseña la cantidad de títulos que son 'movies' o 'show' en cada país de los 10 que hemos seleccionado anteriormente:
        data_1 = x[['type', 'prod_country']].groupby('prod_country')['type'].value_counts().unstack().loc[country]
        # Creamos una nueva columna que suma las dos columnas de 'type' Así podremos sacar el porcentaje:
        data_1['sum'] = data_1.sum(axis=1)
        # Sacamos los porcentajes:
        data_ratio = (data_1.T / data_1['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie')

        # Definimos el tamaño del gráfico:
        fig, ax = plt.subplots(1,1,figsize=(10, 6),)

        # Dibujamos las barras 
        ax.barh(data_ratio.index, data_ratio['Movie'], color='#db0000', alpha=0.8, label='Movie')
        ax.barh(data_ratio.index, data_ratio['TV Show'], left=data_ratio['Movie'], color='grey', alpha=0.8, label='TV Show')

        ax.set_yticklabels(data_ratio.index, fontsize=11, color=color_)

        # Anotación de porcentajes
        for i in data_ratio.index:
                ax.annotate(f"{data_ratio['Movie'][i]*100:.3}%", xy=(data_ratio['Movie'][i]/2, i), va = 'center', ha='center',fontsize=10, color=color_)

        for i in data_ratio.index:
                ax.annotate(f"{data_ratio['TV Show'][i]*100:.3}%", xy=(data_ratio['Movie'][i]+data_ratio['TV Show'][i]/2, i), va = 'center', ha='center',fontsize=10, color=color_)
        
        # Definimos la leyenda
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

        plt.savefig(path_graph + name_file+'.png',transparent=True)

        return plt.show();

    
def rating_plot(x, name_file, color_='white'):
        rating_country = x['prod_country'].value_counts()[:11].index
        data_rating_1 = x[['rating', 'prod_country']].groupby('prod_country')['rating'].value_counts().unstack().loc[rating_country]
        data_rating_1['sum'] = data_rating_1.sum(axis=1)
        data_ratio_1 = (data_rating_1.T / data_rating_1['sum']).T[['Adults', 'Teens', 'Kids', 'Toddlers']].sort_values(by='Adults')

        # Definimos el tamaño del gráfico:
        fig, ax = plt.subplots(1,1,figsize=(10, 8),)


        # Dibujamos las barras 
        ax.barh(data_ratio_1.index, data_ratio_1['Adults'], 
                color='#db0000', alpha=0.8, label='Adults')
        ax.barh(data_ratio_1.index, data_ratio_1['Teens'], left=data_ratio_1['Adults'],
                color='grey', alpha=0.8, label='Teens')
        ax.barh(data_ratio_1.index, data_ratio_1['Kids'], left=data_ratio_1['Adults']+data_ratio_1['Teens'],
                color='#831010', alpha=0.8, label='Kids')
        ax.barh(data_ratio_1.index, data_ratio_1['Toddlers'], left=data_ratio_1['Adults']+data_ratio_1['Teens']+data_ratio_1['Kids'], 
                color='#000000', alpha=0.8, label='Toddlers')


        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticklabels(data_ratio_1.index, fontfamily='serif', fontsize=11, color=color_)
        

        # Definimos la leyenda
        ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))

        plt.savefig(path_graph + name_file+'.png',transparent=True)
        
        return plt.show();