// network contains a VERY basic implementation of a feed forward
// neural network of arbitrary size.  It's basic in that it
// consists of nested lists of floats.  It would be a simple
// matter to abstract this to make use of structs instead.
package main

import (
	"log"
	"math"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func neuronOutput(weights, x []float64) float64 {
	dot, err := utils.Dot(weights, x)
	if err != nil {
		log.Fatalf("error in neuron calc: %e", err)
	}
	return sigmoid(dot)
}

func feedForward(network [][][]float64, data []float64) [][]float64 {
	var outputs [][]float64

	input := data
	for _, layer := range network {
		inputWithBias := append(input, 1.0)

		output := make([]float64, len(layer))
		for i, neuron := range layer {
			output[i] = neuronOutput(neuron, inputWithBias)
		}
		outputs = append(outputs, output)
		input = output
	}

	return outputs
}

func backpropagate(network [][][]float64, data, target []float64) {
	// for now, assume a single hidden layer like the book
	// TODO: generalize to any number of hidden layers
	nLayers := len(network)
	nHidden := nLayers - 1
	outputLayer := nLayers - 1

	forwardResults := feedForward(network, data)
	hiddenOutputs, outputs := forwardResults[:nHidden], forwardResults[outputLayer]

	// calculate directional changes for each output (derivative)
	// based on the difference between the output and the target
	outputDeltas := make([]float64, len(outputs))
	for i, outNeuron := range outputs {
		outputDeltas[i] = outNeuron * (1.0 - outNeuron) * (outNeuron - target[i])
	}

	// adjust weights for the output layer (using last hidden layer)
	for i, outNeuron := range network[outputLayer] {
		for j, hiddenNeuron := range hiddenOutputs[nHidden-1] {
			outNeuron[j] -= outputDeltas[i] * hiddenNeuron
		}
	}

	// back propagate the errors to the hidden layer
	hiddenDeltas := make([]float64, len(outputs))
	for i, hiddenOutput := range hiddenOutputs[0] {
		directionalChange, err := utils.Dot(outputDeltas, utils.GetColumn(network[outputLayer], i))
		if err != nil {
			panic(err)
		}
		hiddenDeltas[i] = hiddenOutput * (1.0 - hiddenOutput) * directionalChange
	}

	// adjust weights for the hidden layer
	for i, hiddenNeuron := range network[0] {
		for j, input := range append(data, 1.0) {
			hiddenNeuron[j] -= hiddenDeltas[i] * input
		}
	}
}
