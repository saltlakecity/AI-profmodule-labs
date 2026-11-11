package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Два обучаемых слоя: Input -> tanh(Hidden) -> Linear(2).
type MLP struct {
	Inputs, Hidden int
	Parameters     []float64
}

func NewMLP(inputs, hidden int, seed uint64) MLP {
	m := MLP{Inputs: inputs, Hidden: hidden, Parameters: make([]float64, hidden*(inputs+1)+2*(hidden+1))}
	rng := rand.New(rand.NewPCG(seed, seed+1))
	for j := 0; j < hidden; j++ {
		for i := 0; i < inputs; i++ {
			m.Parameters[j*(inputs+1)+i] = (2*rng.Float64() - 1) * math.Sqrt(6/float64(inputs+hidden))
		}
	}
	offset := hidden * (inputs + 1)
	for k := 0; k < 2; k++ {
		for j := 0; j < hidden; j++ {
			m.Parameters[offset+k*(hidden+1)+j] = (2*rng.Float64() - 1) * math.Sqrt(6/float64(hidden+2))
		}
	}
	return m
}

func (m MLP) forward(x, hidden []float64) [2]float64 {
	for j := 0; j < m.Hidden; j++ {
		base := j * (m.Inputs + 1)
		z := m.Parameters[base+m.Inputs]
		for i, value := range x {
			z += m.Parameters[base+i] * value
		}
		hidden[j] = math.Tanh(z)
	}
	var output [2]float64
	offset := m.Hidden * (m.Inputs + 1)
	for k := 0; k < 2; k++ {
		base := offset + k*(m.Hidden+1)
		output[k] = m.Parameters[base+m.Hidden]
		for j, value := range hidden {
			output[k] += m.Parameters[base+j] * value
		}
	}
	return output
}

func sign(x float64) float64 {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}

func (m MLP) accumulateGradient(x []float64, target [2]float64, hidden, gradient []float64) float64 {
	output := m.forward(x, hidden)
	var delta [2]float64
	loss := 0.0
	offset := m.Hidden * (m.Inputs + 1)
	for k := 0; k < 2; k++ {
		diff := output[k] - target[k]
		loss += math.Abs(diff) / 2
		delta[k] = sign(diff) / 2
		base := offset + k*(m.Hidden+1)
		for j, value := range hidden {
			gradient[base+j] += delta[k] * value
		}
		gradient[base+m.Hidden] += delta[k]
	}
	for j, value := range hidden {
		dh := 0.0
		for k := 0; k < 2; k++ {
			dh += delta[k] * m.Parameters[offset+k*(m.Hidden+1)+j]
		}
		dh *= 1 - value*value
		base := j * (m.Inputs + 1)
		for i, input := range x {
			gradient[base+i] += dh * input
		}
		gradient[base+m.Inputs] += dh
	}
	return loss
}

func TrainMLP(m *MLP, data []Example, epochs int, rate float64) ([]float64, error) {
	if len(data) == 0 || epochs < 1 || !finite(rate) || rate <= 0 {
		return nil, fmt.Errorf("MLP: нужны выборка, epochs > 0 и rate > 0")
	}
	if m == nil || m.Inputs < 1 || m.Hidden < 1 || len(m.Parameters) != m.Hidden*(m.Inputs+1)+2*(m.Hidden+1) {
		return nil, fmt.Errorf("некорректная архитектура MLP")
	}
	for _, row := range data {
		if len(row.Input) != m.Inputs {
			return nil, fmt.Errorf("MLP: неверное число признаков")
		}
		for _, x := range row.Input {
			if !finite(x) {
				return nil, fmt.Errorf("MLP: неконечный признак")
			}
		}
		if !finite(row.Target[0]) || !finite(row.Target[1]) {
			return nil, fmt.Errorf("MLP: неконечная цель")
		}
	}
	gradient, firstMoment, secondMoment := make([]float64, len(m.Parameters)), make([]float64, len(m.Parameters)), make([]float64, len(m.Parameters))
	hidden := make([]float64, m.Hidden)
	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewPCG(42, 43))
	beta1Power, beta2Power := 1.0, 1.0
	history := make([]float64, 0, epochs)
	const batchSize = 64
	for epoch := 0; epoch < epochs; epoch++ {
		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		for start := 0; start < len(indices); start += batchSize {
			end := min(start+batchSize, len(indices))
			clear(gradient)
			for _, index := range indices[start:end] {
				row := data[index]
				m.accumulateGradient(row.Input, row.Target, hidden, gradient)
			}
			beta1Power *= 0.9
			beta2Power *= 0.999
			for j := range m.Parameters {
				g := gradient[j] / float64(end-start)
				firstMoment[j] = 0.9*firstMoment[j] + 0.1*g
				secondMoment[j] = 0.999*secondMoment[j] + 0.001*g*g
				correctedM := firstMoment[j] / (1 - beta1Power)
				correctedV := secondMoment[j] / (1 - beta2Power)
				m.Parameters[j] -= rate * correctedM / (math.Sqrt(correctedV) + 1e-8)
				if !finite(m.Parameters[j]) {
					return history, fmt.Errorf("MLP: обучение расходится; уменьшите rate")
				}
			}
		}
		loss := 0.0
		for _, row := range data {
			pred := m.forward(row.Input, hidden)
			loss += (math.Abs(pred[0]-row.Target[0]) + math.Abs(pred[1]-row.Target[1])) / (2 * float64(len(data)))
		}
		if !finite(loss) {
			return history, fmt.Errorf("MLP: неконечная ошибка обучения")
		}
		history = append(history, loss)
	}
	return history, nil
}

func predictWeather(m MLP, x []float64, xScaler, yScaler Scaler) ([2]float64, error) {
	if len(x) != m.Inputs || len(xScaler.Mean) != m.Inputs || len(yScaler.Mean) != 2 {
		return [2]float64{}, fmt.Errorf("неверная размерность прогноза")
	}
	p := m.forward(xScaler.transform(x), make([]float64, m.Hidden))
	for k := range p {
		p[k] = p[k]*yScaler.Scale[k] + yScaler.Mean[k]
		if !finite(p[k]) {
			return [2]float64{}, fmt.Errorf("получен неконечный прогноз")
		}
	}
	p[1] = max(0, p[1])
	return p, nil
}
