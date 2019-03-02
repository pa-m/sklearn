set -e
sed -e s/32/64/g basemlp32.go > basemlp64.go
revive

