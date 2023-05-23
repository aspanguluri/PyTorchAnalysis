import math
import random

class OutputNeuron:
	def __init__(self, inputs, bias, learning_rate, target):
		self.inputs = inputs
		self.weights = [random.random() for i in range(0,len(self.inputs))]
		self.bias = bias
		self.learning_rate = learning_rate
		self.target = target

	def get_bias(self):
		return self.bias

	def sigma(self):
		'''
		gets the sum of the inputs and weights
		'''
		s = 0
		for i in range(0,len(self.inputs)):
			s+=(self.inputs[i]*self.weights[i])
		s+=self.bias

		return s

	def sigmoid(self):
		'''
		uses sigmoid function to return the neuron output
		'''

		val = 1/(1 + math.exp(-1*self.sigma()))

		return val

	def get_new_weight(self, index, total_error):
		existing_weight = self.weights[index]

		error_derivative = (self.sigmoid()-self.target) * (self.sigmoid() * (1-self.sigmoid())) * self.inputs[index]

		new_weight = existing_weight-(self.learning_rate * error_derivative)
		
		return new_weight

class HiddenNeuron:
	def __init__(self, inputs, bias, learning_rate):
		self.inputs = inputs
		self.weights = [random.random() for i in range(0,len(self.inputs))]
		self.bias = bias
		self.learning_rate = learning_rate

	def get_bias(self):
		return self.bias

	def sigma(self):
		'''
		gets the sum of the inputs and weights
		'''
		s = 0
		for i in range(0,len(self.inputs)):
			s+=(self.inputs[i]*self.weights[i])
		s+=self.bias

		return s

	def sigmoid(self):
		'''
		uses sigmoid function to return the neuron output
		'''

		val = 1/(1 + math.exp(-1*self.sigma()))

		return val

	def get_new_weight(self, index, total_error, output_1, output_2, target_1, target_2, weights_1, weights_2):

		def Specific_Error_Over_Hidden_Output(inputs, index, output, target, weights):
			
			Error_Output_Over_Out_Output = output - target

			Out_Output_Over_Net_Output = weights[index]

			Error_Output_Over_Net_Output = Error_Output_Over_Out_Output * Out_Output_Over_Net_Output



			Net_Output_Over_Weight = inputs[index]

			return Error_Output_Over_Net_Output * Net_Output_Over_Weight




		existing_weight = self.weights[index]

		Error1_Over_Hidden_Output = Specific_Error_Over_Hidden_Output(self.inputs, index, output_1, target_1, weights_1)

		Error2_Over_Hidden_Output = Specific_Error_Over_Hidden_Output(self.inputs, index, output_2, target_2, weights_2)

		Error_Total_Over_Hidden_Output = Error1_Over_Hidden_Output + Error2_Over_Hidden_Output

		Hidden_Output_Over_Net_Hidden = self.sigmoid() * (1-self.sigmoid())

		Net_Hidden_Over_Weight = self.inputs[index]

		error_derivative = Error_Total_Over_Hidden_Output * Hidden_Output_Over_Net_Hidden * Net_Hidden_Over_Weight

		new_weight = existing_weight-(self.learning_rate * error_derivative)
		
		return new_weight






