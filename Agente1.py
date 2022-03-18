def Simple_Problem_Solving_Agent (percept):
    return "No hagas nada"

iteracion=0
while (iteracion<30):
    accion=Simple_Problem_Solving_Agent(iteracion)
    iteracion+=1
    print("iteracion: ",iteracion," Accion: ",accion)
