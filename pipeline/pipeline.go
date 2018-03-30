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
}

// NewPipeline returns a *Pipeline
func NewPipeline(steps ...NamedStep) *Pipeline {
	p := &Pipeline{NamedSteps: steps}
	return p
}

// Fit for Pipeline
func (p *Pipeline) Fit(X, Y *mat.Dense) *Pipeline {
	Xtmp, Ytmp := X, Y
	for _, step := range p.NamedSteps {
		step.Step.Fit(Xtmp, Ytmp)
		Xtmp, Ytmp = step.Step.Transform(Xtmp, Ytmp)

	}
	return p
}

// Predict ...
func (p *Pipeline) Predict(X, Y *mat.Dense) *Pipeline {
	Xtmp, Ytmp := X, Y
	for _, step := range p.NamedSteps {
		fmt.Println("before ", step.Name, Xtmp.At(0, 0))
		Xtmp, Ytmp = step.Step.Transform(Xtmp, Ytmp)
		fmt.Println("after ", step.Name, Xtmp.At(0, 0))
	}
	for iStep := len(p.NamedSteps) - 2; iStep >= 0; iStep-- {
		step := p.NamedSteps[iStep]
		fmt.Println("before iv ", step.Name, Ytmp.At(0, 0))
		step.Step.(preprocessing.InverseTransformer).InverseTransform(nil, Y)
		fmt.Println("after iv ", step.Name, Ytmp.At(0, 0))
	}
	return p

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
