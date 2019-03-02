package datasets

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"io/ioutil"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// MLDataset structure returned by LoadIris,LoadBreastCancer,LoadDiabetes,LoadBoston
type MLDataset struct {
	Data         [][]float64 `json:"data,omitempty"`
	Target       []float64   `json:"target,omitempty"`
	TargetNames  []string    `json:"target_names,omitempty"`
	DESCR        string      `json:"DESCR,omitempty"`
	FeatureNames []string    `json:"feature_names,omitempty"`
	X, Y         *mat.Dense
}

// fix data path for travis
func realPath(filepath string) string {
	if _, err := os.Stat(filepath); err != nil {
		p := "/home/travis/gopath/src/github.com/"
		lp := len(p)
		if len(filepath) > lp && filepath[0:lp] == p {
			filepath = "/home/travis/build/" + filepath[lp:]
		}
	}
	return filepath
}

// LoadIris load the iris dataset
func loadJSON(filepath string) (ds *MLDataset) {
	dat, err := ioutil.ReadFile(realPath(filepath))
	check(err)
	ds = &MLDataset{}
	err = json.Unmarshal(dat, &ds)
	check(err)
	ds.X, ds.Y = ds.GetXY()
	return
}

// LoadIris load the iris dataset
func LoadIris() (ds *MLDataset) {
	return loadJSON(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/iris.json")
}

// LoadBreastCancer load the breat cancer dataset
func LoadBreastCancer() (ds *MLDataset) {
	return loadJSON(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/cancer.json")
}

// LoadDiabetes load the diabetes dataset
func LoadDiabetes() (ds *MLDataset) {
	return loadJSON(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/diabetes.json")
}

// LoadBoston load the boston housing dataset
func LoadBoston() (ds *MLDataset) {
	return loadJSON(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/boston.json")
}

// GetXY returns X,Y matrices for iris dataset
func (ds *MLDataset) GetXY() (X, Y *mat.Dense) {
	nSamples, nFeatures, nOutputs := len(ds.Data), len(ds.FeatureNames), 1
	X = mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, _ float64) float64 {
		return ds.Data[i][j]
	}, X)
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, _ int, _ float64) float64 {
		return ds.Target[i]
	}, Y)
	return
}

// LoadExamScore loads data from ex2data1 from Andrew Ng machine learning course
func LoadExamScore() (X, Y *mat.Dense) {
	return loadCsv(os.Getenv("GOPATH")+"/src/github.com/pa-m/sklearn/datasets/data/ex2data1.txt", nil, 1)

}

// LoadMicroChipTest loads data from ex2data2 from  Andrew Ng machine learning course
func LoadMicroChipTest() (X, Y *mat.Dense) {
	return loadCsv(os.Getenv("GOPATH")+"/src/github.com/pa-m/sklearn/datasets/data/ex2data2.txt", nil, 1)
}

// LoadMnist loads mnist data 5000x400,5000x1
func LoadMnist() (X, Y *mat.Dense) {
	mats := LoadOctaveBin(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/ex4data1.dat.gz")
	return mats["X"], mats["y"]
}

// LoadMnistWeights loads mnist weights
func LoadMnistWeights() (Theta1, Theta2 *mat.Dense) {
	mats := LoadOctaveBin(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/ex4weights.dat.gz")
	return mats["Theta1"], mats["Theta2"]
}

func loadCsv(filepath string, setupReader func(*csv.Reader), nOutputs int) (X, Y *mat.Dense) {
	f, err := os.Open(realPath(filepath))
	check(err)
	defer f.Close()
	r := csv.NewReader(f)
	if setupReader != nil {
		setupReader(r)
	}
	cells, err := r.ReadAll()
	check(err)
	nSamples, nFeatures := len(cells), len(cells[0])-nOutputs
	X = mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, _ float64) float64 { x, err := strconv.ParseFloat(cells[i][j], 64); check(err); return x }, X)
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, o int, _ float64) float64 {
		y, err := strconv.ParseFloat(cells[i][nFeatures], 64)
		check(err)
		return y
	}, Y)
	return
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// LoadInternationalAirlinesPassengers ...
func LoadInternationalAirlinesPassengers() (Y *mat.Dense) {
	f, err := os.Open(realPath(os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/international-airline-passengers.csv"))
	check(err)
	defer f.Close()
	fb := bufio.NewReader(f)
	fb.ReadLine()
	r := csv.NewReader(fb)
	r.Comma = ','

	cells, err := r.ReadAll()
	check(err)
	nSamples, nOutputs := len(cells), 1
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, o int, _ float64) float64 {
		y, err := strconv.ParseFloat(cells[i][1], 64)
		check(err)
		return y
	}, Y)
	return
}
