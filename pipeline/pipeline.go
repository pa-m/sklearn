package pipeline

import (
	"fmt"
	"strings"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"

	"gonum.org/v1/gonum/mat"
)

// Estimator is an interface for Predict
type Estimator interface {
	Predict(X, Y *mat.Dense)
}

// NamedStep represents a pipeline named Step
type NamedStep struct {
	Name string
	Step base.Transformer
}

// Pipeline is a sequance of transformers and an estimator
type Pipeline struct {
	NamedSteps []NamedStep
	NOutputs   int
}

// NewPipeline returns a *Pipeline
func NewPipeline(steps ...NamedStep) *Pipeline {
	p := &Pipeline{NamedSteps: steps}
	return p
}

// Fit for Pipeline
func (p *Pipeline) Fit(X, Y *mat.Dense) base.Transformer {
	_, p.NOutputs = Y.Dims()
	Xtmp, Ytmp := X, Y
	for _, step := range p.NamedSteps {
		step.Step.Fit(Xtmp, Ytmp)
		Xtmp, Ytmp = step.Step.Transform(Xtmp, Ytmp)

	}
	return p
}

// Predict ...
func (p *Pipeline) Predict(X, Y *mat.Dense) base.Regressor {
	Xtmp, Ytmp := X, Y
	for _, step := range p.NamedSteps {
		Xtmp, Ytmp = step.Step.Transform(Xtmp, Ytmp)

	}
	for iStep := len(p.NamedSteps) - 2; iStep >= 0; iStep-- {
		step := p.NamedSteps[iStep]
		_, Ytmp = step.Step.(preprocessing.InverseTransformer).InverseTransform(nil, Ytmp)

	}
	Y.Copy(Ytmp)
	return p

}

// Transform for pipeline
func (p *Pipeline) Transform(X, Y *mat.Dense) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	Xout = X
	Yout = mat.NewDense(nSamples, p.NOutputs, nil)
	p.Predict(Xout, Yout)
	return
}

// Score for base.Regressor
func (p *Pipeline) Score(X, Y *mat.Dense) float64 {
	Xtmp, Ytmp := X, Y
	for _, step := range p.NamedSteps[:len(p.NamedSteps)-1] {
		Xtmp, Ytmp = step.Step.Transform(Xtmp, Ytmp)

	}
	return p.NamedSteps[len(p.NamedSteps)-1].Step.(base.Regressor).Score(Xtmp, Y)
}

// MakePipeline returns a Pipeline from unnamed steps
func MakePipeline(steps ...base.Transformer) *Pipeline {
	p := &Pipeline{}
	for _, step := range steps {
		/*if _, ok := step.(preprocessing.Transformer); (istep < len(steps)-1) && !ok {
			panic(fmt.Errorf("%T is not a Transformer", step))
		}
		if _, ok := step.(lm.Regressor); (istep == len(steps)-1) && !ok {
			panic(fmt.Errorf("%T is not a lm.Regressor", step))
		}*/

		p.NamedSteps = append(p.NamedSteps, NamedStep{Name: strings.ToLower(fmt.Sprintf("%T", step)), Step: step})
	}
	return p
}
