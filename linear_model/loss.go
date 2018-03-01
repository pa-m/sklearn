package linearModel

// Hinge loss , for SVMs, h=-1 +1. v https://en.wikipedia.org/wiki/Hinge_loss
// F: math.Max(0.,1.-h*y)
// Fprime: if 1. > h*y{return -y*hprime}else {return 0.}

// Squared Loss, Quadratic Loss, for regressions
// F: mat.Pow(h-y,2)/2
// Fprime: h hprime- y hprime //==> hprime*(h-y)

// Cross entropy Loss
// F: -y*math.Log(h)-(1.-y)*log(1.-h)
// Fprime: -y hprime/h - (1-y) * (-hprime)/(1-h)
// Fprime: (h-y)/(1.-h)* hprime/h
