import Lib_squats as lbs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# boolean = True
# mpampis = 0.2
# string = '2'
# integer = 2
# integer_2 = 4
# list = [2,3,"hola", 0.2]
# tuple = (2,3,"hola", 0.2)
# dictionary = {'Holla': 2, 'Better': 4}
#
# mpampis, string = 0.2, '2'
# lbs.white_noise_signal_creation_using_FFT_method(100)
# def vrasidas(a,b,c):
#     result = a + b - c
#     return result
#
# kostas = vrasidas(mpampis,integer, integer_2)
# print(kostas)
#
# j = 0
# k = 3
# o = 14
# second_kosts = vrasidas(j,k,o)
# print(second_kosts)



# if mpampis < 5:
#     print("well done")
# else:
#     print("skata")
#
#     helo = 'helo'
#     print(helo)
# #
# list_integer = []
# for item in list:
#     print(item)
#     if isinstance(item, int):
#         list_integer.append(item)
#
# print(list_integer)
#
# i = 0
# smoking = False
# while smoking == False:
#     if i <= 5:
#         i = i + 1
#     else:
#         smoking = True
#         print(smoking)
#         print(i)

data = pd.read_excel('Data.xlsx', header=None)
print(data)
plt.plot(data['x coordinates'], label='x')
plt.plot(data['y coordinates'], label='y')
plt.legend()
plt.show()