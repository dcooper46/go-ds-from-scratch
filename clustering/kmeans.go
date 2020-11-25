package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Basic implementation of Kmeans clustering aglorithm

// KMeans represents a k-means model defined
// by the number of clusters and mean vector of each
type KMeans struct {
	k     int
	means [][]float64
}

// Classify takes an input feature vector and returns the cluster
// from the trained KMeans model nearest to it
func (km *KMeans) Classify(x []float64) int {
	var cluster int
	minDist := math.Inf(1)
	for c, mu := range km.means {
		muDist, err := utils.SquaredDistance(x, mu)
		if err != nil {
			log.Fatal(err)
		}
		if muDist < minDist {
			minDist = muDist
			cluster = c
		}
	}
	return cluster
}

// Train takes input training data and determines the optimal
// clusters based on the number requested
func (km *KMeans) Train(data [][]float64) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// initialize cluster centers with random elements from data
	km.means = make([][]float64, km.k)
	for i := 0; i < km.k; i++ {
		// mu := make([]float64, dim)
		// for j := 0; j < dim; j++ {
		// 	mu[j] = rand.Float64()
		// }
		km.means[i] = data[r.Intn(len(data))]
	}
	fmt.Printf("initial means: %v\n", km.means)

	clusters := make([]float64, len(data))
	newClusters := make([]float64, len(data))
	for {
		// classify each record using current model means
		for i, di := range data {
			newClusters[i] = float64(km.Classify(di))
		}

		// if no change to assignments, done
		if utils.VectorsEqual(clusters, newClusters) {
			break
		}

		clusters = newClusters

		// update model means
		for c := 0; c < km.k; c++ {
			var cData [][]float64
			for i, ci := range clusters {
				if ci == float64(c) {
					cData = append(cData, data[i])
				}
			}
			if len(cData) > 0 {
				cmeans, err := utils.ColumnMeans(cData)
				if err != nil {
					log.Fatal(err)
				}
				km.means[c] = cmeans
			}
		}
	}
}
