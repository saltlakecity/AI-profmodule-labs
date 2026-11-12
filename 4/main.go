package main

import (
	"bytes"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

//go:embed data/moscow.csv
var weatherCSV []byte

func checkOutputPath(dataPath, resultPath string) error {
	if dataPath == "" {
		return nil
	}
	source, err := os.Stat(dataPath)
	if err != nil {
		return fmt.Errorf("проверка входного файла: %w", err)
	}
	target, err := os.Stat(resultPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("проверка выходного файла: %w", err)
	}
	if os.SameFile(source, target) {
		return fmt.Errorf("выходной файл совпадает с входным датасетом; выберите другую папку -out")
	}
	return nil
}

func run() error {
	dataPath := flag.String("data", "", "CSV: date,temperature,precipitation")
	epochs := flag.Int("epochs", 120, "число эпох")
	rate := flag.Float64("rate", 0.003, "шаг обучения")
	out := flag.String("out", "results", "папка результата")
	flag.Parse()
	if *epochs < 1 || !finite(*rate) || *rate <= 0 {
		return fmt.Errorf("нужны epochs > 0 и rate > 0")
	}
	resultPath := filepath.Join(*out, "result.txt")
	if err := checkOutputPath(*dataPath, resultPath); err != nil {
		return err
	}
	content := weatherCSV
	var err error
	if *dataPath != "" {
		content, err = os.ReadFile(*dataPath)
		if err != nil {
			return err
		}
	}
	days, err := readWeather(bytes.NewReader(content))
	if err != nil {
		return err
	}
	data := makeExamples(days)
	splitAt := len(data) * 4 / 5
	train, test := data[:splitAt], data[splitAt:]
	scaled, xScaler, yScaler := normalizeTraining(train)
	model := NewMLP(len(train[0].Input), 16, 42)
	if err := TrainMLP(&model, scaled, *epochs, *rate); err != nil {
		return err
	}
	var mae [2]float64
	for _, row := range test {
		prediction, err := predictWeather(model, row.Input, xScaler, yScaler)
		if err != nil {
			return err
		}
		for k := range mae {
			mae[k] += math.Abs(prediction[k]-row.Target[k]) / float64(len(test))
		}
	}
	tomorrow := days[len(days)-1].Date.AddDate(0, 0, 1)
	forecast, err := predictWeather(model, features(days[len(days)-lookback:], tomorrow), xScaler, yScaler)
	if err != nil {
		return err
	}
	report := fmt.Sprintf("ЛАБОРАТОРНАЯ №4. НЕЙРОСЕТЕВОЙ ПРОГНОЗ ПОГОДЫ\n\nОбучение: %s — %s (%d объектов)\nТест: %s — %s (%d объектов)\nMAE температуры = %.4f °C\nMAE осадков = %.4f мм/сутки\n\nПрогноз на %s\nСредняя температура = %.2f °C\nСуммарные осадки = %.2f мм/сутки\n", train[0].Date.Format("2006-01-02"), train[len(train)-1].Date.Format("2006-01-02"), len(train), test[0].Date.Format("2006-01-02"), test[len(test)-1].Date.Format("2006-01-02"), len(test), mae[0], mae[1], tomorrow.Format("2006-01-02"), forecast[0], forecast[1])
	if bytes.Equal(content, weatherCSV) {
		report += "\nДанные: Москва, ВДНХ (станция 27612); отобраны три столбца, значения не изменены.\nИсточник: https://www.kaggle.com/datasets/geopod/russian-weather-1980-2023\n© 2025 Tkachenko M.A., Fomin D.S.: https://doi.org/10.5281/zenodo.17789545\nЛицензия: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/\n"
	} else {
		report += "\nИсточник данных: " + *dataPath + "\n"
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	if err := os.WriteFile(resultPath, []byte(report), 0644); err != nil {
		return err
	}
	fmt.Print(report)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
