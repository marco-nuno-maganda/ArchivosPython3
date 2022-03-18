import time
import os
from datetime import datetime

# datetime object containing current date and time
Remitente="MarcoNuño"
Parte1='mosquitto_pub -h 173.82.206.12 -t /house/light -m '
RetardoEnSegundos=1

NumeroMensajes=4
for i in range (1,NumeroMensajes+1):
    now = datetime.now() 
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    Mensaje="'"+Remitente+":"+dt_string+"'"
    os.system(Parte1+Mensaje)
    time.sleep(RetardoEnSegundos) # Sleep for N seconds
