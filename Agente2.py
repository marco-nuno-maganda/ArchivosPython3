class Agente:
    def __init__(self):       
        self.seq=[]
        self.state=0
        self.goal="null"
        self.problem=""
                
    def Simple_Problem_Solving_Agent (self, percept):
        self.state=self.Update_state(self.state,percept)
        
        # Preguntart si Self.Seq esta vacio?
        print(len(self.seq))
        if len(self.seq)==0:
            print ("Lista Vacia")
            self.goal = self.Formulate_Goal(self.state)
            self.problem = self.Formulate_Problem(self.state,self.goal)
            self.seq=self.Search(self.problem)
            if self.seq is None: # Failure (return null action)
                self.seq=[]
                return (" Fallo")
        action=self.seq[0]
        print (self.seq)
        self.seq.pop(0)
        print (self.seq)
        return (action)
         
        #self.seq.append("No hagas nada")
        #return "No hagas nada"
        

    def Formulate_Goal(self,  state):
        return ""            
    def Formulate_Problem(self, state, goal):
        return "Y"
    def Search(self, problem):
        # Regrese "Failure" , Regrese Una lista de acciones
        if problem=="X":
            return None
        else:         
            return ["Accion1", "Accion2", "Accion3", "Accion4"]
        #
        #return ["Accion1"]      


    def Update_state (self,state,percept):
        # Conjunto de If-ELSE dependiento del nuevo estado del sistema en base a 
        #la percepcion
        return self.state +1 # El nuevo estado del sistema es un independientemente delo que percibe    

A1=Agente()
iteracion=0
while(iteracion<30):
    action=A1.Simple_Problem_Solving_Agent (1)
    iteracion+=1
    print ("Iteracion",iteracion," Accion:",action, "Estado", A1.state)

print(A1.seq)
