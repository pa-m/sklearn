// Package neuralnetwork reproduces multilayer perceptron
// based on Andreas Mueller implementation
// + float 32 implementation
// + weight decay
// + batch normalization
package neuralnetwork

//go:generate sh -c "sed -e s/32/64/g basemlp32.go > basemlp64.go"
