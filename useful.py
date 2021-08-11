import pandas as pd
import numpy as np
import random as rnd


# Вычисляет дисперсию вектор-столбца
def dispersion(x):
    z = len(x) - 1
    x = x - x.values.mean()
    return np.sum(np.square(x)) / z


# Генератор паролей
def password_gen(lenth=16):
    key_list = '0123456789' + \
               'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ans1_list = ['да', 'Да', 'ок', 'Ок', 'Yes', 'yes', 'y']
    password = ''
    t = 0

    ans1 = str(input('Do you need specific password lenth? '))

    if ans1 not in ans1_list:
        lenth = lenth

    elif t == 3:
        print('Maybe you have problems with input')
        lenth = lenth

    else:
        ans2 = str(input('Input password lenth you need: '))
        while ans2.isdigit() != True:
            print('\nYour input is incorrect. Input int number.')
            ans2 = str(input('Input password lenth you need: '))
            t += 1

        lenth = int(ans2)

    for i in range(lenth):

        if i == lenth // 3 or i == (lenth // 3) * 2:
            password += '-'
        else:
            password += key_list[rnd.randint(0, len(key_list) - 1)]
    return password, print(password)

password_gen()