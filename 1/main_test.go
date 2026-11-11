package main

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestPerceptronLearnsAND(t *testing.T) {
	data := []Sample{{[]float64{0, 0}, 0}, {[]float64{0, 1}, 0}, {[]float64{1, 0}, 0}, {[]float64{1, 1}, 1}}
	p, history, err := Train(data, 0.1, 100)
	if err != nil || history[len(history)-1] != 0 {
		t.Fatalf("обучение: %v, %v", history, err)
	}
	for _, sample := range data {
		got, err := p.Predict(sample.X)
		if err != nil || got != sample.Y {
			t.Fatalf("Predict(%v) = %d, %v; want %d", sample.X, got, err, sample.Y)
		}
	}
}

func TestNonseparableAndInvalidData(t *testing.T) {
	xor := []Sample{{[]float64{0, 0}, 0}, {[]float64{0, 1}, 1}, {[]float64{1, 0}, 1}, {[]float64{1, 1}, 0}}
	_, history, err := Train(xor, 0.1, 20)
	if err == nil || len(history) != 20 {
		t.Fatalf("XOR должен достигнуть лимита: %v", err)
	}
	for _, data := range [][]Sample{nil, {{[]float64{1}, 2}}, {{[]float64{math.NaN()}, 0}}, {{[]float64{1}, 0}, {[]float64{1, 2}, 1}}} {
		if _, _, err := Train(data, 0.1, 10); err == nil {
			t.Fatalf("приняты некорректные данные %v", data)
		}
	}
}

func TestCSVAndEmbeddedExample(t *testing.T) {
	data, _, err := readCSV(bytes.NewReader(trainingCSV), true, nil)
	if err != nil {
		t.Fatal(err)
	}
	p, _, err := Train(data, 0.1, 1000)
	if err != nil {
		t.Fatal(err)
	}
	for _, point := range []Sample{{[]float64{2, 3}, 1}, {[]float64{-2, -1}, 0}, {[]float64{3, 1}, 1}} {
		got, err := p.Predict(point.X)
		if err != nil || got != point.Y {
			t.Fatalf("%v: %d, %v", point.X, got, err)
		}
	}
	for _, input := range []string{"x,class\n1,2\n", "x,class\nNaN,1\n", "x,class\n1\n", "x,class\n", "class\n1\n"} {
		if _, _, err := readCSV(strings.NewReader(input), true, nil); err == nil {
			t.Fatalf("принят некорректный CSV %q", input)
		}
	}
}

func TestCSVRequiresTrainingFeatureOrder(t *testing.T) {
	_, features, err := readCSV(strings.NewReader("x1,x2,class\n2,-2,1\n"), true, nil)
	if err != nil {
		t.Fatal(err)
	}
	data, _, err := readCSV(strings.NewReader("x1,x2\n2,-2\n"), false, features)
	if err != nil || len(data) != 1 || data[0].X[0] != 2 || data[0].X[1] != -2 {
		t.Fatalf("совпадающая схема: %v, %v", data, err)
	}
	for _, input := range []string{"x2,x1\n-2,2\n", "other,x2\n2,-2\n", "x1\n2\n"} {
		if _, _, err := readCSV(strings.NewReader(input), false, features); err == nil {
			t.Fatalf("принята несовпадающая схема %q", input)
		}
	}
}
