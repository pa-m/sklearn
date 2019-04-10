package pipeline

import (
	"fmt"

	// "log"
	"strings"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"

	"gonum.org/v1/gonum/mat"
)

// NamedStep represents a pipeline named Step
// Step must be Predicter (last step) or Transformer
type NamedStep struct {
	Name string
	base.Fiter
}

// Pipeline is a sequance of transformers and an estimator
type Pipeline struct {
	NamedSteps []NamedStep

	NOutputs int
}

// NewPipeline returns a *Pipeline
func NewPipeline(steps ...NamedStep) *Pipeline {
	p := &Pipeline{NamedSteps: steps}
	return p
}

// PredicterClone for pipeline relies on children clone method is child is Transformer
func (p *Pipeline) PredicterClone() base.Predicter {
	clone := *p
	clone.NamedSteps = make([]NamedStep, len(p.NamedSteps))
	for i, step := range p.NamedSteps {
		if cloner, ok := step.Fiter.(base.Transformer); ok {
			clone.NamedSteps[i] = NamedStep{Name: step.Name, Fiter: cloner.TransformerClone()}
		} else if cloner, ok := step.Fiter.(base.Predicter); ok {
			clone.NamedSteps[i] = NamedStep{Name: step.Name, Fiter: cloner.PredicterClone()}
		} else {
			panic(fmt.Errorf("step %s is not clonable", step.Name))
		}
	}
	return &clone
}

// IsClassifier for pipeline returns last step IsClassifier if any
func (p *Pipeline) IsClassifier() bool {
	nSteps := len(p.NamedSteps)
	if nSteps > 0 {
		if clf, ok := p.NamedSteps[nSteps-1].Fiter.(base.Predicter); ok {
			return clf.IsClassifier()
		}
	}
	return false
}

func (p *Pipeline) transformStep(istep int, Xtmp, Ytmp **mat.Dense) {
	step := p.NamedSteps[istep]
	if transformer, ok := step.Fiter.(base.Transformer); ok {
		*Xtmp, *Ytmp = transformer.Transform(*Xtmp, *Ytmp)
	} else if predicter, ok := step.Fiter.(base.Predicter); ok {
		nOutputs := predicter.GetNOutputs()
		if nOutputs == 0 {
			panic(fmt.Errorf("pipeline step %d (%s) predicter with no output", istep, step.Name))
		}
		nSamples, _ := (*Xtmp).Dims()
		*Ytmp = mat.NewDense(nSamples, nOutputs, nil)
		predicter.Predict(*Xtmp, *Ytmp)
	} else {
		if istep < len(p.NamedSteps)-1 {
			panic(fmt.Errorf("pipeline step %d (%s) is not a Transformer", istep, step.Name))
		} else {
			panic(fmt.Errorf("pipeline step %d (%s) is not a Predicter", istep, step.Name))
		}
	}

}

// Fit for Pipeline
func (p *Pipeline) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	_, p.NOutputs = Y.Dims()
	Xtmp, Ytmp := X, Y
	steps := len(p.NamedSteps)
	for istep, step := range p.NamedSteps {
		step.Fit(Xtmp, Ytmp)
		if istep < steps-1 {
			p.transformStep(istep, &Xtmp, &Ytmp)
		}
	}
	return p
}

// Score for pipeline
func (p *Pipeline) Score(X, Y mat.Matrix) float64 {
	Xtmp, Ytmp := base.ToDense(X), base.ToDense(Y)
	for istep := range p.NamedSteps[:len(p.NamedSteps)-1] {
		p.transformStep(istep, &Xtmp, &Ytmp)

	}
	last := p.NamedSteps[len(p.NamedSteps)-1]
	predicter, ok := last.Fiter.(base.Predicter)
	if !ok {
		panic(fmt.Errorf("pipeline.Score: last step is not a Predicter"))
	}
	return predicter.Score(Xtmp, Y)
}

// GetNOutputs returns output columns number for Y
func (p *Pipeline) GetNOutputs() int {
	if len(p.NamedSteps) == 0 {
		return 0
	}
	return p.NamedSteps[len(p.NamedSteps)-1].Fiter.(base.Predicter).GetNOutputs()
}

// Predict for pipeline calls steps Transform then InverseTransform for InverseTransformer steps
func (p *Pipeline) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	Xtmp, Ytmp := base.ToDense(X), base.ToDense(Y)
	for istep := range p.NamedSteps {
		p.transformStep(istep, &Xtmp, &Ytmp)

	}
	for iStep := len(p.NamedSteps) - 2; iStep >= 0; iStep-- {
		step := p.NamedSteps[iStep]
		_, Ytmp = step.Fiter.(preprocessing.InverseTransformer).InverseTransform(nil, Ytmp)
	}
	return base.FromDense(Y, base.ToDense(Ytmp))
}

// Transform for pipeline
func (p *Pipeline) Transform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	nSamples, _ := X.Dims()
	Xout = base.ToDense(X)
	Yout = mat.NewDense(nSamples, p.NOutputs, nil)
	p.Predict(Xout, Yout)
	return
}

// FitTransform fit to dat, then transform it
func (p *Pipeline) FitTransform(X, Y mat.Matrix) (Xout, Yout *mat.Dense) {
	p.Fit(X, Y)
	return p.Transform(X, Y)
}

// MakePipeline returns a Pipeline from unnamed steps
func MakePipeline(steps ...base.Fiter) *Pipeline {
	p := &Pipeline{}
	for _, step := range steps {
		p.NamedSteps = append(p.NamedSteps, NamedStep{Name: strings.ToLower(fmt.Sprintf("%T", step)), Fiter: step})
	}
	return p
}
