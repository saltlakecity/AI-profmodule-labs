package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

const lookback = 7

type Weather struct {
	Date                       time.Time
	Temperature, Precipitation float64
}

type Example struct {
	Date     time.Time
	Input    []float64
	Target   [2]float64
	Previous [2]float64
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func readWeather(r io.Reader) ([]Weather, error) {
	reader := csv.NewReader(r)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("заголовок погоды: %w", err)
	}
	if len(header) != 3 || strings.TrimPrefix(header[0], "\ufeff") != "date" || header[1] != "temperature" || header[2] != "precipitation" {
		return nil, fmt.Errorf("ожидается CSV: date,temperature,precipitation")
	}
	var days []Weather
	for line := 2; ; line++ {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("строка %d: %w", line, err)
		}
		date, err := time.Parse("2006-01-02", strings.TrimSpace(row[0]))
		if err != nil {
			return nil, fmt.Errorf("строка %d: дата должна иметь формат YYYY-MM-DD", line)
		}
		temp, err := strconv.ParseFloat(strings.TrimSpace(row[1]), 64)
		if err != nil || !finite(temp) {
			return nil, fmt.Errorf("строка %d: некорректная температура", line)
		}
		precip, err := strconv.ParseFloat(strings.TrimSpace(row[2]), 64)
		if err != nil || !finite(precip) || precip < 0 {
			return nil, fmt.Errorf("строка %d: осадки должны быть конечным неотрицательным числом", line)
		}
		days = append(days, Weather{date, temp, precip})
	}
	if len(days) < lookback+10 {
		return nil, fmt.Errorf("нужно хотя бы %d ежедневных наблюдений", lookback+10)
	}
	sort.Slice(days, func(i, j int) bool { return days[i].Date.Before(days[j].Date) })
	for i := 1; i < len(days); i++ {
		expected := days[i-1].Date.AddDate(0, 0, 1)
		if !days[i].Date.Equal(expected) {
			return nil, fmt.Errorf("после %s ожидается %s, получено %s: пропуск или дубликат даты", days[i-1].Date.Format("2006-01-02"), expected.Format("2006-01-02"), days[i].Date.Format("2006-01-02"))
		}
	}
	return days, nil
}

func features(history []Weather, targetDate time.Time) []float64 {
	x := make([]float64, 0, lookback*2+2)
	for _, day := range history {
		x = append(x, day.Temperature, day.Precipitation)
	}
	daysInYear := time.Date(targetDate.Year()+1, 1, 1, 0, 0, 0, 0, time.UTC).Sub(time.Date(targetDate.Year(), 1, 1, 0, 0, 0, 0, time.UTC)).Hours() / 24
	phase := 2 * math.Pi * float64(targetDate.YearDay()-1) / daysInYear
	return append(x, math.Sin(phase), math.Cos(phase))
}

func makeExamples(days []Weather) []Example {
	data := make([]Example, 0, len(days)-lookback)
	for i := lookback; i < len(days); i++ {
		data = append(data, Example{
			Date:     days[i].Date,
			Input:    features(days[i-lookback:i], days[i].Date),
			Target:   [2]float64{days[i].Temperature, days[i].Precipitation},
			Previous: [2]float64{days[i-1].Temperature, days[i-1].Precipitation},
		})
	}
	return data
}

type Scaler struct{ Mean, Scale []float64 }

func fitScaler(rows [][]float64) Scaler {
	s := Scaler{make([]float64, len(rows[0])), make([]float64, len(rows[0]))}
	for _, row := range rows {
		for j, x := range row {
			s.Mean[j] += x / float64(len(rows))
		}
	}
	for _, row := range rows {
		for j, x := range row {
			d := x - s.Mean[j]
			s.Scale[j] += d * d / float64(len(rows))
		}
	}
	for j := range s.Scale {
		s.Scale[j] = math.Sqrt(s.Scale[j])
		if s.Scale[j] == 0 {
			s.Scale[j] = 1
		}
	}
	return s
}

func (s Scaler) transform(row []float64) []float64 {
	x := make([]float64, len(row))
	for j, v := range row {
		x[j] = (v - s.Mean[j]) / s.Scale[j]
	}
	return x
}

func normalizeTraining(data []Example) ([]Example, Scaler, Scaler) {
	inputs, targets := make([][]float64, len(data)), make([][]float64, len(data))
	for i, row := range data {
		inputs[i] = row.Input
		targets[i] = []float64{row.Target[0], row.Target[1]}
	}
	xScaler, yScaler := fitScaler(inputs), fitScaler(targets)
	scaled := make([]Example, len(data))
	for i, row := range data {
		scaled[i] = row
		scaled[i].Input = xScaler.transform(row.Input)
		y := yScaler.transform(targets[i])
		scaled[i].Target = [2]float64{y[0], y[1]}
	}
	return scaled, xScaler, yScaler
}
