__author__ = 'felip'

import string
import random
import csv






resultado = ['']*1500002
entrada = input()
z = 0
f = 0



for y in range(0, 15000, 1):
    f = f+1
    for x in range(0, 100, 1):
        z = z + 1
        saida = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(entrada))
        resultado[z] = str(f), saida


c = csv.writer(open("senhas.csv", "wb"))
for row in resultado:
    c.writerow(row)









