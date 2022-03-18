import re

#re.findall(r'\b[a-zA-Z]{4}\b',line) for line in open('/home/marco/Escritorio/69_LenguajesAutomatas/Lista_LA.txt')]

with open("/home/marco/Escritorio/69_LenguajesAutomatas/Lista_LA.txt") as f:
    pattern =r"\b[A-Z]{4}\b"
    for line in f:
        print (line)           
        #result = regex.search(line)
        #lista=re.findall(pattern,line)
        #print (lista)
        result = re.match(pattern, line)
        print (result)

exit()

import re


#pattern = '^a...s$'
#pattern = '^j[a-z]*e$'
#pattern = '^j[a-z]*$'
#pattern = '^j..e$'
#pattern = 'j..e$'
#pattern = 'j[a-z]*e'
#pattern = 'abc|12'
# 3 letras o dos digitos (cualquiera)

#pattern = '^...|12'
#pattern ="\d{2}|[a-zA-Z]{3}$"
#pattern ="\d{2}|w{3}"
#pattern ="\d{2}|[a-zA-Z]{3}\b"
#pattern ="\d{2}|\b[a-zA-Z]{3}\b"
#pattern =r"d{2}"
#pattern1 =r"\b\d{2}\b|\b[a-zA-Z]{3}\b"
#pattern1 =r"\b\d{4}\b"
#pattern1 =r"\d+"
#pattern1 ="[a-zA-Z0–9._-]+@[a-zA-Z0–9._-]+\.[a-zA-Z]{2,4}"
pattern1 ="[a-zA-Z0–9._-]+"
pattern1 ="[a-zA-Z0-9._-]+"
pattern1 ="[a-zA-Z0-9._-]+.[a-zA-Z0-9._-]+"
pattern1 ="[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,4}"

print (pattern1)

test_string = 'josue'
test_string = 'jose'
test_string = 'jonas'
test_string = 'z'
test_string = 'josue jose'
test_string = 'abcd efg 12 hij 345 klm no pq rst wyz 6789'
test_string = 'mnunom_upv@upv.edu.mx'
#test_string = '178454'
#test_string = 'upv.edu.mx'
#result = re.match(pattern1, test_string)
result = re.findall(pattern1, test_string)
print (result)

if result:
  print("Search successful.")
else:
  print("Search unsuccessful.")
