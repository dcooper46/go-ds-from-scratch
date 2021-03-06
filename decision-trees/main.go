package main

import "fmt"

var data = []map[string]string{
	{"level": "Senior", "lang": "Java", "tweets": "no", "phd": "no", "label": "false"},
	{"level": "Senior", "lang": "Java", "tweets": "no", "phd": "yes", "label": "false"},
	{"level": "Mid", "lang": "Python", "tweets": "no", "phd": "no", "label": "true"},
	{"level": "Junior", "lang": "Python", "tweets": "no", "phd": "no", "label": "true"},
	{"level": "Junior", "lang": "R", "tweets": "yes", "phd": "no", "label": "true"},
	{"level": "Junior", "lang": "R", "tweets": "yes", "phd": "yes", "label": "false"},
	{"level": "Mid", "lang": "R", "tweets": "yes", "phd": "yes", "label": "true"},
	{"level": "Senior", "lang": "Python", "tweets": "no", "phd": "no", "label": "false"},
	{"level": "Senior", "lang": "R", "tweets": "yes", "phd": "no", "label": "true"},
	{"level": "Junior", "lang": "Python", "tweets": "yes", "phd": "no", "label": "true"},
	{"level": "Senior", "lang": "Python", "tweets": "yes", "phd": "yes", "label": "true"},
	{"level": "Mid", "lang": "Python", "tweets": "no", "phd": "yes", "label": "true"},
	{"level": "Mid", "lang": "Java", "tweets": "yes", "phd": "no", "label": "true"},
	{"level": "Junior", "lang": "Python", "tweets": "no", "phd": "yes", "label": "false"},
}

func main() {
	for _, attribute := range []string{"level", "lang", "tweets", "phd"} {
		fmt.Printf("%s: %f\n", attribute, GetPartitionEntropy(data, attribute))
	}

	tree := BuildTreeID3(data, []string{"level", "lang", "tweets", "phd"})

	tree.Show()

	fmt.Printf("Junior / Java / tweets / no phd: %v\n\n", tree.Classify(map[string]string{
		"level":  "Junior",
		"lang":   "Java",
		"tweets": "Yes",
		"phd":    "no",
	}))

	fmt.Printf("Junior / Java / tweets / phd: %v\n\n", tree.Classify(map[string]string{
		"level":  "Junior",
		"lang":   "Java",
		"tweets": "Yes",
		"phd":    "yes",
	}))

	fmt.Printf("Intern: %v\n\n", tree.Classify(map[string]string{
		"level": "Intern",
	}))

	fmt.Printf("Senior: %v\n", tree.Classify(map[string]string{
		"level": "Senior",
	}))

}
