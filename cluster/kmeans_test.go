package cluster

import (
	"fmt"
	"image/color"
	"os"
	"os/exec"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var (
	_ base.Predicter = &KMeans{}
)

func ExampleKMeans() {
	ds := datasets.LoadIris()
	X := ds.X
	NSamples, _ := X.Dims()
	Y := mat.NewDense(NSamples, 1, nil)
	kmeans := &KMeans{NClusters: 3}
	start := time.Now()
	_ = start
	kmeans.Fit(X, nil)
	kmeans.Predict(X, Y)
	//fmt.Printf("elapsed %s s\n", time.Since(start))

	// v https://github.com/gonum/plot/wiki/Example-plots
	if *visualDebug {
		xplot, yplot := 0, 2
		getData := func(value int) (scatterData plotter.XYs) {
			for i := 0; i < NSamples; i++ {
				if int(Y.At(i, 0)) == value {
					scatterData = append(scatterData, struct{ X, Y float64 }{X.At(i, xplot), X.At(i, yplot)})
				}
			}
			return
		}
		p, err := plot.New()
		if err != nil {
			panic(err)
		}
		p.Title.Text = "kmeans"
		p.X.Label.Text = "X"
		p.Y.Label.Text = "Y"
		// Draw a grid behind the data
		p.Add(plotter.NewGrid())
		for cl := 0; cl < kmeans.NClusters; cl++ {
			s, err := plotter.NewScatter(getData(cl))
			if err != nil {
				panic(err)
			}
			var color0 color.RGBA
			switch cl {
			case 0:
				color0 = color.RGBA{176, 0, 0, 255}
			case 1:
				color0 = color.RGBA{0, 176, 0, 255}
			case 2:
				color0 = color.RGBA{0, 0, 176, 255}
			}
			s.GlyphStyle.Color = color0
			p.Add(s)
			p.Legend.Add(fmt.Sprintf("scatter %d", cl), s)
			{
				c := kmeans.Centroids.RowView(cl)
				sc, err := plotter.NewScatter(plotter.XYs{{X: c.AtVec(xplot), Y: c.AtVec(yplot)}})
				if err != nil {
					panic(err)
				}
				sc.GlyphStyle.Shape = draw.PlusGlyph{}
				sc.GlyphStyle.Color = color0
				p.Add(sc)
			}

		}
		// Save the plot to a PNG file.
		pngfile := "/tmp/plt.png"
		os.Remove(pngfile)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, pngfile); err != nil {
			panic(err)
		}
		cmd := exec.Command("display", pngfile)
		err = cmd.Start()
		if err != nil {
			fmt.Println(err.Error())
		}
		time.Sleep(200 * time.Millisecond)
		os.Remove(pngfile)
	}
	// Output:
}
