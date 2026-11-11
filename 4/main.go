package main

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

//go:embed data/moscow.csv
var weatherCSV []byte

const sourceURL = "https://www.kaggle.com/datasets/geopod/russian-weather-1980-2023"

func checkOutputPaths(dataPath, outputDir string) error {
	if dataPath == "" {
		return nil
	}
	source, err := os.Stat(dataPath)
	if err != nil {
		return fmt.Errorf("проверка входного файла: %w", err)
	}
	for _, name := range []string{"test_predictions.csv", "training.csv", "forecast.csv", "model.json", "report.md"} {
		path := filepath.Join(outputDir, name)
		target, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("проверка выходного файла %s: %w", path, err)
		}
		if os.SameFile(source, target) {
			return fmt.Errorf("выходной файл %s совпадает с входным датасетом; выберите другую папку -out", path)
		}
	}
	return nil
}

func writeCSV(path string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	err = w.WriteAll(rows)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func run() error {
	dataPath := flag.String("data", "", "CSV: date,temperature,precipitation (по умолчанию Москва, ВДНХ, Kaggle)")
	epochs := flag.Int("epochs", 120, "число эпох обучения MLP")
	rate := flag.Float64("rate", 0.003, "шаг Adam")
	out := flag.String("out", "results", "папка отчёта, прогнозов и весов")
	flag.Parse()
	if *epochs < 1 || !finite(*rate) || *rate <= 0 {
		return fmt.Errorf("нужны epochs > 0 и rate > 0")
	}
	if err := checkOutputPaths(*dataPath, *out); err != nil {
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
	datasetName := "Москва (ВДНХ), станция 27612"
	source := sourceURL
	if *dataPath != "" {
		datasetName = "Пользовательский CSV"
		source = *dataPath
	}
	fmt.Printf("ЛАБОРАТОРНАЯ №4. НЕЙРОСЕТЕВОЙ ПРОГНОЗ ПОГОДЫ\nДанные: %s\nИсточник: %s\nНаблюдений: %d; период: %s — %s\nПризнаки: температура и осадки за %d дней, sin/cos дня года\nАрхитектура: %d -> %d (tanh) -> 2 (линейный выход), обучаемых параметров: %d\nОбучение: %d примеров, цели %s — %s\nТест: %d примеров, цели %s — %s\nПорядок разбиения: первые 80%% / последние 20%% по времени\nНормализация обучена только на train; тест не участвует в обновлении весов\nОптимизатор: Adam; rate = %g; эпох: %d; batch = 64; seed = 42\nФункция потерь: MAE нормализованных целей\n", datasetName, source, len(days), days[0].Date.Format("2006-01-02"), days[len(days)-1].Date.Format("2006-01-02"), lookback, model.Inputs, model.Hidden, len(model.Parameters), len(train), train[0].Date.Format("2006-01-02"), train[len(train)-1].Date.Format("2006-01-02"), len(test), test[0].Date.Format("2006-01-02"), test[len(test)-1].Date.Format("2006-01-02"), *rate, *epochs)
	started := time.Now()
	loss, err := TrainMLP(&model, scaled, *epochs, *rate)
	elapsed := time.Since(started)
	if err != nil {
		return err
	}
	for i, value := range loss {
		if i == 0 || (i+1)%10 == 0 || i == len(loss)-1 {
			fmt.Printf("Эпоха %d: train MAE (норм.) = %.6f\n", i+1, value)
		}
	}
	rows := [][]string{{"date", "temperature_actual", "temperature_predicted", "precipitation_actual", "precipitation_predicted", "temperature_persistence", "precipitation_persistence"}}
	var mae, persistence [2]float64
	zeroRainMAE := 0.0
	for _, row := range test {
		prediction, err := predictWeather(model, row.Input, xScaler, yScaler)
		if err != nil {
			return err
		}
		for k := range mae {
			mae[k] += math.Abs(prediction[k]-row.Target[k]) / float64(len(test))
			persistence[k] += math.Abs(row.Previous[k]-row.Target[k]) / float64(len(test))
		}
		zeroRainMAE += row.Target[1] / float64(len(test))
		rows = append(rows, []string{row.Date.Format("2006-01-02"), fmt.Sprintf("%.6f", row.Target[0]), fmt.Sprintf("%.6f", prediction[0]), fmt.Sprintf("%.6f", row.Target[1]), fmt.Sprintf("%.6f", prediction[1]), fmt.Sprintf("%.6f", row.Previous[0]), fmt.Sprintf("%.6f", row.Previous[1])})
	}
	last := days[len(days)-1]
	tomorrow := last.Date.AddDate(0, 0, 1)
	forecast, err := predictWeather(model, features(days[len(days)-lookback:], tomorrow), xScaler, yScaler)
	if err != nil {
		return err
	}
	fmt.Printf("\nОЦЕНКА НА ТЕСТЕ\nMAE температуры = %.4f °C\nMAE осадков = %.4f мм/сутки\nБазовый прогноз «как вчера»: MAE = %.4f °C и %.4f мм/сутки\nБазовый прогноз «осадков нет»: MAE = %.4f мм/сутки\nВремя обучения: %.3f s\n\nПРОГНОЗ НА СЛЕДУЮЩИЙ ДЕНЬ ДАТАСЕТА\nПоследние данные: %s\nДата прогноза: %s\nСредняя температура: %.2f °C\nСуммарные осадки: %.2f мм/сутки\n", mae[0], mae[1], persistence[0], persistence[1], zeroRainMAE, elapsed.Seconds(), last.Date.Format("2006-01-02"), tomorrow.Format("2006-01-02"), forecast[0], forecast[1])
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	if err := writeCSV(filepath.Join(*out, "test_predictions.csv"), rows); err != nil {
		return err
	}
	lossRows := [][]string{{"epoch", "train_mae_normalized"}}
	for i, value := range loss {
		lossRows = append(lossRows, []string{fmt.Sprint(i + 1), fmt.Sprintf("%.10f", value)})
	}
	if err := writeCSV(filepath.Join(*out, "training.csv"), lossRows); err != nil {
		return err
	}
	if err := writeCSV(filepath.Join(*out, "forecast.csv"), [][]string{{"date", "temperature", "precipitation"}, {tomorrow.Format("2006-01-02"), fmt.Sprintf("%.6f", forecast[0]), fmt.Sprintf("%.6f", forecast[1])}}); err != nil {
		return err
	}
	saved := struct {
		Model                     MLP
		InputScaler, OutputScaler Scaler
		Lookback                  int
		TrainedThrough            string
		Source                    string
	}{model, xScaler, yScaler, lookback, train[len(train)-1].Date.Format("2006-01-02"), source}
	encoded, err := json.MarshalIndent(saved, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*out, "model.json"), append(encoded, '\n'), 0644); err != nil {
		return err
	}
	var report strings.Builder
	fmt.Fprintf(&report, "# Лабораторная работа №4. Нейросетевой прогноз погоды\n\n## Данные\n\n%s. Источник: %s.\n\nСуточная средняя температура (°C) и суточная сумма осадков (мм), %d строк: %s — %s.\n", datasetName, source, len(days), days[0].Date.Format("2006-01-02"), last.Date.Format("2006-01-02"))
	if *dataPath == "" {
		report.WriteString("\nГотовый датасет RusWeather-GF, авторы: Tkachenko M.A., Fomin D.S. (2025), DOI: https://doi.org/10.5281/zenodo.17789545. Лицензия CC BY 4.0. Из версии Kaggle №2 отобрана станция 27612, оставлены три столбца, значения не изменены. Датасет уже содержит восстановленные авторами пропуски. По описанию источника сутки определяются в UTC.\n")
	}
	fmt.Fprintf(&report, "\n## Метод\n\nДвухслойный MLP написан с нуля на Go: %d входов → %d нейронов tanh → 2 линейных выхода. Два слоя имеют обучаемые веса; входной слой в это число не входит. Признаки: температура и осадки за последние %d суток, синус и косинус дня года прогнозируемой даты. Календарь известен заранее; погода целевого дня не используется во входе.\n\nПервые 80%% примеров — train (%d, %s — %s), последние 20%% — test (%d, %s — %s). Средние и стандартные отклонения признаков и целей рассчитаны только по train. Перемешиваются только обучающие примеры внутри эпох. Тестовые даты не участвуют в обучении, подборе числа эпох или нормализации.\n\nОбратное распространение вычисляет субградиент MAE двух нормализованных целей. Обновление весов: Adam, rate = %g, %d эпох, batch = 64, seed = 42. Выходы возвращаются в исходные единицы; отрицательный прогноз осадков заменяется нулём.\n", model.Inputs, model.Hidden, lookback, len(train), train[0].Date.Format("2006-01-02"), train[len(train)-1].Date.Format("2006-01-02"), len(test), test[0].Date.Format("2006-01-02"), test[len(test)-1].Date.Format("2006-01-02"), *rate, *epochs)
	fmt.Fprintf(&report, "\n## Качество\n\nMAE = sum(|прогноз − факт|) / N. Температура и осадки оцениваются отдельно, поскольку имеют разные единицы.\n\n| Модель | MAE температуры, °C | MAE осадков, мм/сутки |\n|---|---:|---:|\n| MLP | %.4f | %.4f |\n| Как вчера | %.4f | %.4f |\n| Нулевые осадки | — | %.4f |\n\nОбучение заняло %.3f s. Train MAE в нормализованных единицах: %.6f после первой эпохи, %.6f после последней.\n\nТест — последовательность однодневных прогнозов: для каждого дня доступны фактические наблюдения предыдущих семи дней, включая уже наступившие тестовые дни. Веса при тестировании не обновляются. Это не один прогноз сразу на весь тестовый период.\n", mae[0], mae[1], persistence[0], persistence[1], zeroRainMAE, elapsed.Seconds(), loss[0], loss[len(loss)-1])
	fmt.Fprintf(&report, "\n## Прогноз на следующий день\n\nПоследняя дата данных: **%s**. Прогнозируемая дата: **%s**.\n\n- Средняя температура: **%.2f °C**.\n- Суммарные осадки: **%.2f мм/сутки**.\n\nПрогноз выполнен той же моделью, которая оценивалась на test. После оценки модель не дообучалась на тестовых целях. «Завтра» означает следующий день исторического датасета, как согласовано в задании.\n\n## Ограничения\n\nЭто учебная модель по локальной истории температуры и осадков: она не использует атмосферные поля и не заменяет метеорологический прогноз. Оценка относится к одному временному разбиению. Исходный RusWeather-GF уже содержит статистически восстановленные значения; их влияние нельзя отделить без исходных флагов качества. Нормализация и обучение в этой программе используют только train, но восстановление пропусков выполнено авторами датасета заранее.\n\nПолные тестовые прогнозы: `test_predictions.csv`; история обучения: `training.csv`; итоговый прогноз: `forecast.csv`; веса и нормализация: `model.json`.\n", last.Date.Format("2006-01-02"), tomorrow.Format("2006-01-02"), forecast[0], forecast[1])
	if err := os.WriteFile(filepath.Join(*out, "report.md"), []byte(report.String()), 0644); err != nil {
		return err
	}
	fmt.Printf("Отчёт, CSV и веса: %s\n", *out)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
