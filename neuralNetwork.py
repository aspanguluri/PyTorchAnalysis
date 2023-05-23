from neuron import OutputNeuron, HiddenNeuron
import random

neuron_inputs_Set1 = [0.0, 1.0]
neuron_inputs_Set2 = [0, 1.0]

neuron_inputs = [neuron_inputs_Set1, neuron_inputs_Set2]

output_neuron_1_targets = [1.0, 0.0]
output_neuron_2_targets = [0.0, 1.0]


hidden_neuron_1 = HiddenNeuron(neuron_inputs_Set1, 0, 0.01)
hidden_neuron_2 = HiddenNeuron(neuron_inputs_Set1, 0, 0.01)

output_neuron_1 = OutputNeuron([hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()], 0, 0.01, 1.0)
output_neuron_2 = OutputNeuron([hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()], 0, 0.01, 0.0)

#training the data
epochs = 100000
for _ in range(0,epochs):
	for i in range(0,2):
		output_neuron_1.target = output_neuron_1_targets[i]
		output_neuron_2.target = output_neuron_2_targets[i]

		hidden_neuron_1.inputs = neuron_inputs[i]
		hidden_neuron_2.inputs = neuron_inputs[i]


		#outputs of the output neurons
		output_neuron_1.inputs = [hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()]
		output_neuron_2.inputs = [hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()]

		out_output_neuron_1 = output_neuron_1.sigmoid()
		out_output_neuron_2 = output_neuron_2.sigmoid()

		print(out_output_neuron_1, out_output_neuron_2)

		#error of each output neuron
		error_1 = 0.5 * (output_neuron_1.target - out_output_neuron_1)**2
		error_2 = 0.5 * (output_neuron_2.target - out_output_neuron_2)**2

		#total error
		error_total = error_1 + error_2

		new_weights = []

		new_weights_o1 = []
		for i in range(0,len(output_neuron_1.inputs)):
			new_weights_o1.append(output_neuron_1.get_new_weight(i, error_total))
		new_weights.append(new_weights_o1)

		new_weights_o2 = []
		for i in range(0,len(output_neuron_2.inputs)):
			new_weights_o2.append(output_neuron_2.get_new_weight(i, error_total))
		new_weights.append(new_weights_o2)

		new_weights_h1 = []
		for i in range(0,len(hidden_neuron_1.inputs)):
			new_weights_h1.append(hidden_neuron_1.get_new_weight(i, error_total, out_output_neuron_1, out_output_neuron_2, output_neuron_1.target, output_neuron_2.target, output_neuron_1.weights, output_neuron_2.weights))
		new_weights.append(new_weights_h1)

		new_weights_h2 = []
		for i in range(0,len(hidden_neuron_2.inputs)):
			new_weights_h2.append(hidden_neuron_2.get_new_weight(i, error_total, out_output_neuron_1, out_output_neuron_2, output_neuron_1.target, output_neuron_2.target, output_neuron_1.weights, output_neuron_2.weights))
		new_weights.append(new_weights_h2)


		#reassigning weights
		for i in range(0, len(new_weights_o1)):
			output_neuron_1.weights[i] = new_weights_o1[i]

		for i in range(0, len(new_weights_o2)):
			output_neuron_2.weights[i] = new_weights_o2[i]

		for i in range(0, len(new_weights_h1)):
			hidden_neuron_1.weights[i] = new_weights_h1[i]

		for i in range(0, len(new_weights_h2)):
			hidden_neuron_2.weights[i] = new_weights_h2[i]

hidden_neuron_1.inputs = neuron_inputs_Set1
hidden_neuron_2.inputs = neuron_inputs_Set1

output_neuron_1.inputs = [hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()]
output_neuron_2.inputs = [hidden_neuron_1.sigmoid(), hidden_neuron_2.sigmoid()]

print(output_neuron_1.sigmoid())
print(output_neuron_2.sigmoid())