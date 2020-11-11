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

func backpropagate(network [][][]float64, data, target []float64) [][]float64 {
	hiddenOutputs, outputs = feedForward
}
