package main

import "fmt"

func main() {
	xorNetwork := [][][]float64{
		{
			{20.0, 20.0, -30.0}, // 'and' neuron
			{20.0, 20.0, -10.0}, // 'or' neuron
		},
		{{-60.0, 60.0, -30.0}}, // '2nd input but not first input' neuron
	}

	for _, x := range []float64{0.0, 1.0} {
		for _, y := range []float64{0.0, 1.0} {
			fmt.Printf("%f, %f, %v\n",
				x,
				y,
				feedForward(xorNetwork, []float64{x, y})[1],
			)
		}
	}
}
