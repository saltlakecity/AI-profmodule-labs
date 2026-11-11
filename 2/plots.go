package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

type functionGrid struct {
	objective              Objective
	xmin, xmax, ymin, ymax float64
}

func (g functionGrid) Dims() (int, int)   { return 100, 100 }
func (g functionGrid) X(c int) float64    { return g.xmin + float64(c)*(g.xmax-g.xmin)/99 }
func (g functionGrid) Y(r int) float64    { return g.ymin + float64(r)*(g.ymax-g.ymin)/99 }
func (g functionGrid) Z(c, r int) float64 { return g.objective.Value(Point{g.X(c), g.Y(r)}) }

func saveTrajectory(path string, steps []DescentStep) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	rows := [][]string{{"iteration", "x", "y", "value", "gradient_norm", "alpha"}}
	for i, s := range steps {
		rows = append(rows, []string{strconv.Itoa(i), fmt.Sprintf("%.12g", s.X), fmt.Sprintf("%.12g", s.Y), fmt.Sprintf("%.12g", s.Value), fmt.Sprintf("%.12g", s.GradNorm), fmt.Sprintf("%.12g", s.Alpha)})
	}
	err = w.WriteAll(rows)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func savePlot(path string, objective Objective, trajectories [][]DescentStep, names []string) error {
	p := plot.New()
	p.Title.Text = "f(x,y) = " + objective.Name
	p.X.Label.Text, p.Y.Label.Text = "x", "y"
	p.Legend.Top = true
	p.Legend.Left = true
	xmin, xmax, ymin, ymax := objective.Minimum.X-1, objective.Minimum.X+1, objective.Minimum.Y-1, objective.Minimum.Y+1
	for _, steps := range trajectories {
		for _, s := range steps {
			xmin, xmax = min(xmin, s.X-0.5), max(xmax, s.X+0.5)
			ymin, ymax = min(ymin, s.Y-0.5), max(ymax, s.Y+0.5)
		}
	}
	grid := functionGrid{objective, xmin, xmax, ymin, ymax}
	fmin := objective.Value(objective.Minimum)
	contours := plotter.NewContour(grid, []float64{fmin + 0.1, fmin + 0.5, fmin + 1, fmin + 3, fmin + 8, fmin + 20, fmin + 50, fmin + 100}, nil)
	contours.LineStyles = []draw.LineStyle{{Color: color.Gray{Y: 180}, Width: vg.Points(0.6)}}
	p.Add(contours)
	colors := []color.Color{color.RGBA{R: 32, G: 99, B: 195, A: 255}, color.RGBA{R: 210, G: 65, B: 35, A: 255}}
	for i, steps := range trajectories {
		points := make(plotter.XYs, len(steps))
		for j, s := range steps {
			points[j] = plotter.XY{X: s.X, Y: s.Y}
		}
		line, dots, err := plotter.NewLinePoints(points)
		if err != nil {
			return err
		}
		line.Color, line.Width = colors[i], vg.Points(1.7)
		dots.Color, dots.Radius = colors[i], vg.Points(2)
		p.Add(line, dots)
		p.Legend.Add(names[i], line, dots)
	}
	for i, point := range []Point{trajectories[0][0].Point, objective.Minimum} {
		s, err := plotter.NewScatter(plotter.XYs{{X: point.X, Y: point.Y}})
		if err != nil {
			return err
		}
		s.Radius = vg.Points(5)
		label := "Start"
		s.Shape, s.Color = draw.CircleGlyph{}, color.Black
		if i == 1 {
			label, s.Shape, s.Color = "Minimum", draw.CrossGlyph{}, color.RGBA{G: 140, A: 255}
		}
		p.Add(s)
		p.Legend.Add(label, s)
	}
	p.X.Min, p.X.Max, p.Y.Min, p.Y.Max = xmin, xmax, ymin, ymax
	return p.Save(20*vg.Centimeter, 16*vg.Centimeter, path)
}
