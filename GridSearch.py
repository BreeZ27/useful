
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def grid_search(df=df[['x_train']], y=df.y_train, ex=3, df_need=False):
    # Функция работает по модели линейной регрессии. Определяет влияние степени полинома
    # одной переменной на качество аппроксимации по показателю R2.

    # На вход принимает вектор-столбец Х - df, вектор столбец У - y,
    # максимальную степень полинома для исследования - ex, булево значние (False по уполчанию), при установке True
    # в качестве 0 элемента кортежа передаст массив со столбцами (модель, значения аппркс. функции, r2_score)

    # Выход функции по умолчанию - набор графиков для полинома каждой степени, сообщение с указанием степени полинома,
    # R2 которого максимален.

    new_df = pd.DataFrame()
    ind_list = []

    models_df = pd.DataFrame(columns=['model', 'predict', 'r2_score'])

    fig, ax = plt.subplots(ncols=2, nrows=ex // 2 + 1, figsize=(10, 15))

    col = 0

    for i in range(1, ex + 1):

        new_df['e' + str(i)] = df[df.columns[0]] ** i

        models_df = models_df.append({'model': LinearRegression().fit(new_df[new_df.columns.tolist()], y)}, \
                                     ignore_index=True)
        models_df.at[i - 1, 'predict'] = models_df['model'][i - 1].predict(new_df[new_df.columns.tolist()])
        models_df.at[i - 1, 'r2_score'] = r2_score(y, models_df['predict'][i - 1])

        if i == ex and ex % 2 != 0:
            ax1 = plt.subplot2grid(shape=(ex // 2 + 1, 2), loc=(ex // 2, 0), colspan=2)

            ax1.scatter(x=df, y=y)
            ax1.plot(df, models_df['predict'][i - 1], color='red')
            ax1.set_title('Approximation function for polynomial of degree ' + str(i) + '\n' + \
                          'r2_score = ' + str(round(models_df.r2_score[i - 1], 5)))
            ax1.set_xlabel('x values')
            ax1.set_ylabel('y values')
            ax1.grid()

        if i <= ex // 2:

            ax[i - 1, col].scatter(x=df, y=y)
            ax[i - 1, col].plot(df, models_df['predict'][i - 1], color='red')
            ax[i - 1, col].set_title('Approximation function for polynomial of degree ' + str(i) + '\n' + \
                                     'r2_score = ' + str(round(models_df.r2_score[i - 1], 5)))
            ax[i - 1, col].set_xlabel('x values')
            ax[i - 1, col].set_ylabel('y values')
            ax[i - 1, col].grid()

        else:
            if col < 1:
                col += 1

            ax[i - ex // 2 - 1, col].scatter(x=df, y=y)
            ax[i - ex // 2 - 1, col].plot(df, models_df['predict'][i - 1], color='red')
            ax[i - ex // 2 - 1, col].set_title('Approximation function for polynomial of degree ' + str(i) + '\n' + \
                                               'r2_score = ' + str(round(models_df.r2_score[i - 1], 6)))
            ax[i - ex // 2 - 1, col].set_xlabel('x values')
            ax[i - ex // 2 - 1, col].set_ylabel('y values')
            ax[i - ex // 2 - 1, col].grid()

        ind_list.append('e' + str(i))

    if ex % 2 == 0:
        for k in range(2):
            fig.delaxes(ax[ex // 2, k])

    models_df.index = ind_list
    plt.tight_layout()
    plt.show()

    if df_need == False:
        return 'The best approximation of the data by a polynomial of degree ' + \
               str(models_df[models_df['r2_score'] == models_df.r2_score.max()].index[0][1:])
    else:
        return models_df, 'The best approximation of the data by a polynomial of degree ' + \
               str(models_df.r2_score.max())


grid_search(ex=10)