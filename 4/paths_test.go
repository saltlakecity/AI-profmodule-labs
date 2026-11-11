package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOutputPathsRejectInputCollision(t *testing.T) {
	for _, name := range []string{"test_predictions.csv", "training.csv", "forecast.csv", "model.json", "report.md"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, name)
			if err := os.WriteFile(path, []byte(weatherFixture()), 0600); err != nil {
				t.Fatal(err)
			}
			if err := checkOutputPaths(path, dir); err == nil {
				t.Fatal("разрешена перезапись входного датасета")
			}
		})
	}
}

func TestOutputPathsRecognizeHardLinks(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "weather.csv")
	if err := os.WriteFile(source, []byte(weatherFixture()), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(source, filepath.Join(dir, "forecast.csv")); err != nil {
		t.Skipf("жёсткие ссылки недоступны: %v", err)
	}
	if err := checkOutputPaths(source, dir); err == nil {
		t.Fatal("разрешена перезапись датасета через жёсткую ссылку")
	}
}

func TestOutputPathsAllowSeparateDataset(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "weather.csv")
	if err := os.WriteFile(source, []byte(weatherFixture()), 0600); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{dir, filepath.Join(dir, "new-results")} {
		if err := checkOutputPaths(source, output); err != nil {
			t.Fatal(err)
		}
	}
	if err := checkOutputPaths("", dir); err != nil {
		t.Fatal(err)
	}
	if err := checkOutputPaths(filepath.Join(dir, "missing.csv"), dir); err == nil {
		t.Fatal("не обнаружен отсутствующий входной файл")
	}
}
