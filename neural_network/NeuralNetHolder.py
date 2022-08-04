import csv
import math


#==================================================================================================================

class Neuron():
    def __init__(self, Activation_Value, Num_of_Weights, Neuron_Index):
        
        self.AV = Activation_Value
        self.Weights = []
        self.Neuron_Index = Neuron_Index
        
        for j in range(Num_of_Weights):
            self.Weights.append(0)



    def Multi_Weights(self, Prev_Layer):  # Calculating Values for next Following Neuron

        result = 0

        for j in range(len(Prev_Layer)):
            result = Prev_Layer[j].AV * Prev_Layer[j].Weights[self.Neuron_Index] + result

        return 1 / (1 + math.exp(-result))

#==================================================================================================================

Input_Layer  = [Neuron(0, 4, 0), Neuron(0, 4, 1),Neuron(1, 4, 2)]
Hidden_Layer = [Neuron(0, 2, 0), Neuron(0, 2, 1),Neuron(0, 2, 2),Neuron(0, 2, 3),Neuron(1, 2, 4)]
Output_Layer = [Neuron(0, 0, 0), Neuron(0, 0, 1)]


#==================================================================================================================

def FeedForward_process(inputs):
    for i in range(len(Input_Layer) - 1):
        Input_Layer[i].AV = inputs[i]

    for i in range(len(Hidden_Layer) - 1):
        Hidden_Layer[i].AV = Hidden_Layer[i].Multi_Weights(Input_Layer)

    for i in range(len(Output_Layer)):
        Output_Layer[i].AV = Output_Layer[i].Multi_Weights(Hidden_Layer)

    Feed_Forward_Output = []

    for i in range(len(Output_Layer)):
        Feed_Forward_Output.append(Output_Layer[i].AV)

    return Feed_Forward_Output

#==================================================================================================================

with open('Final_Weights.txt') as f:
    for i in range(len(Input_Layer)):
        lines = f.readline()
        #print(lines)
        inputweights=lines.split(",")
        #print(inputweights)
        Input_Layer[i].weights=inputweights[:2]
        #print(inputLayer[i].weights)
    lines = f.readline()
    lines = f.readline()
    for i in range(len(Hidden_Layer)):
        lines = f.readline()
        #print(lines)
        hiddenweights=lines.split(",")
        #print(hiddenweights)
        Hidden_Layer[i].weights=hiddenweights[:2]
        #print(hiddenLayer[i].weights)

minmax = [[-802.9329157873535, 65.09457405564024, -5.237908992470128, -7.73643352785449], [804.9093491239987, 803.9409319059207, 7.999999999999988, 7.351782318721565]]

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        # this pass can be removed once you add some code
        inputs=input_row.split(",")
        inputs[0]=float(inputs[0])
        inputs[1]=float(inputs[1])
        inputs[0] = (inputs[0] - minmax[0][0]) / (minmax[1][0] - minmax[0][0])
        inputs[1] = (inputs[1] - minmax[0][1]) / (minmax[1][1] - minmax[0][1])

        outputs=FeedForward_process(inputs)


        outputs[0]=(outputs[0]*(minmax[1][2] - minmax[0][2]))+minmax[0][2]
        outputs[1]=(outputs[1]*(minmax[1][3] - minmax[0][3]))+minmax[0][3]


        return outputs
