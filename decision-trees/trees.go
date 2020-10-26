package main

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Entropy measures the uncertainty in data
// represented as a probability distribution
func Entropy(probs []float64) (entropy float64) {
	for _, p := range probs {
		if p > 0.0 {
			entropy += -p * math.Log2(p)
		}
	}
	return
}

// GetProbabilities returns label probabilities given a slice of labels (strings)
func GetProbabilities(labels []string) []float64 {
	n := float64(len(labels))
	labelCounts := make(map[string]int)
	for _, label := range labels {
		labelCounts[label]++
	}
	counts := make([]float64, len(labelCounts))
	var i int
	for _, c := range labelCounts {
		counts[i] = float64(c)
		i++
	}
	return utils.ScalarMultiply(1.0/n, counts)
}

// DataEntropy returns the entropy for a set of labels/data
func DataEntropy(data []string) float64 {
	probs := GetProbabilities(data)
	return Entropy(probs)
}

// PartitionEntropy finds the entropy of a partition of subsets of data
// where the partition entropy is a weighted sum of the entropies of
// each subset
//
// H = q1*H(S1) + q2*H(S2) + ... + qn*H(Sn) where qi is the subset proportion
func PartitionEntropy(subsets map[string][]map[string]string) float64 {
	var N float64
	var PartialH []float64

	for _, s := range subsets {
		N += float64(len(s))
		vals := make([]string, len(s))
		for i, v := range s {
			vals[i] = v["label"]
		}
		PartialH = append(PartialH, DataEntropy(vals)*float64(len(vals)))
	}

	return utils.VectorSum(PartialH) / N
}

// PartitionBy groups data by values of an attribute
//
// here it is assumed the desired label is the last element in each record
func PartitionBy(data []map[string]string, attribute string) map[string][]map[string]string {
	groups := make(map[string][]map[string]string)
	for _, row := range data {
		groups[row[attribute]] = append(groups[row[attribute]], row)
	}
	return groups
}

// GetPartitionEntropy returns the entropy of data partitioned by a specific attribute
func GetPartitionEntropy(data []map[string]string, attribute string) float64 {
	partitions := PartitionBy(data, attribute)
	return PartitionEntropy(partitions)
}

// BoolTree represents a binary decision tree
type BoolTree struct {
	Attr  string
	Leaf  bool
	Value bool
	Nodes map[string]*BoolTree
}

// BuildTreeID3 buils a  binary decision tree using the ID3 algorithm
func BuildTreeID3(data []map[string]string, splitAttributes []string) *BoolTree {

	// count trues and falses from the input data
	NTrue, NFalse := 0, 0
	for _, dmap := range data {
		label, err := strconv.ParseBool(dmap["label"])
		if err != nil {
			log.Fatalf("bad label: %e", err)
		}
		if label {
			NTrue++
		} else {
			NFalse++
		}
	}

	if NTrue == 0 {
		return &BoolTree{
			Leaf:  true,
			Value: false,
		}
	}
	if NFalse == 0 {
		return &BoolTree{
			Leaf:  true,
			Value: true,
		}
	}
	if len(splitAttributes) == 0 {
		return &BoolTree{
			Leaf:  true,
			Value: NTrue >= NFalse,
		}
	}

	bestAttrEntropy := math.Inf(1)
	var bestAttr string
	for _, attr := range splitAttributes {
		attrEntropy := GetPartitionEntropy(data, attr)
		if attrEntropy < bestAttrEntropy {
			bestAttrEntropy = attrEntropy
			bestAttr = attr
		}
	}

	partitions := PartitionBy(data, bestAttr)
	var newSplitAttributes []string
	for _, attr := range splitAttributes {
		if attr != bestAttr {
			newSplitAttributes = append(newSplitAttributes, attr)
		}
	}

	children := make(map[string]*BoolTree)
	for attr, subset := range partitions {
		children[attr] = BuildTreeID3(subset, newSplitAttributes)
	}
	children["_default"] = &BoolTree{
		Leaf:  true,
		Value: NTrue > NFalse,
	}

	return &BoolTree{
		Attr:  bestAttr,
		Nodes: children,
	}
}

// Classify returns a boolean label for a given input map
// by parsing the defined BoolTree
func (tree *BoolTree) Classify(input map[string]string) bool {
	for {
		if tree.Leaf {
			return tree.Value
		}

		attr := tree.Attr
		node := tree.Nodes

		nodeKey, ok := input[attr]
		if !ok { // tree attribute note in input data
			nodeKey = "_default"
		}

		children, ok := node[nodeKey]
		if !ok {
			fmt.Printf("key not found: %s\n", nodeKey)
			nodeKey = "_default"
			children = node[nodeKey]
		}

		tree = children
	}
}

// Show recursively prints a BoolTree
func (tree *BoolTree) Show(tabs ...string) {
	fmt.Println(strings.Join(append(tabs, "{"), ""))
	fmt.Printf("\t"+strings.Join(tabs, "")+"Attr: %s\n", tree.Attr)
	fmt.Printf("\t"+strings.Join(tabs, "")+"Leaf: %v\n", tree.Leaf)
	fmt.Printf("\t"+strings.Join(tabs, "")+"Value: %v\n", tree.Value)
	fmt.Println("\t" + strings.Join(tabs, "") + "Children: ")
	for a, t := range tree.Nodes {
		fmt.Printf("\t"+strings.Join(tabs, "")+"Label: %s\n", a)
		t.Show(strings.Join(append(tabs, "\t"), ""))
	}
	fmt.Println(strings.Join(append(tabs, "}"), ""))
}
