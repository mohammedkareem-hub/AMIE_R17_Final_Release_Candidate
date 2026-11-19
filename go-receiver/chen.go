package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
)

type chenParams struct {
	initial                     [4]float64
	a, b, c, gamma, d, alpha, h float64
}

func chenDefaults() chenParams {
	return chenParams{initial: [4]float64{0.3, 0.3, 0.3, 0.3}, a: 35, b: 3, c: 12, gamma: 28, d: .5, alpha: .97, h: .001}
}
func chenDerived(masterKey, nonce []byte) (chenParams, error) {
	p := chenDefaults()
	raw, err := derivedKey(masterKey, nonce, "chaotic-initial-conditions")
	if err != nil {
		return p, err
	}
	for i := 0; i < 4; i++ {
		u := float64(binary.BigEndian.Uint64(raw[i*8:])) / math.Exp2(64)
		p.initial[i] += (u - .5) * 1e-6
	}
	return p, nil
}
func saltBytes(s string) []byte { return []byte(s) }
func chenSalted(p chenParams, salt []byte) chenParams {
	h := sha256.Sum256(append([]byte("FractionalChen4D-Salt-v1|"), salt...))
	for i := 0; i < 4; i++ {
		u := float64(binary.BigEndian.Uint64(h[i*8:])) / float64(^uint64(0))
		p.initial[i] += 1e-8 * (u - .5)
	}
	return p
}
func chenSystem(p chenParams, s [4]float64) [4]float64 {
	x, y, z, u := s[0], s[1], s[2], s[3]
	return [4]float64{p.a*(y-x) + u, p.gamma*x - x*z + p.c*y, x*y - p.b*z, y*z + p.d*u}
}
func chenCtx(p chenParams, salt []byte) [32]byte {
	vals := []float64{p.initial[0], p.initial[1], p.initial[2], p.initial[3], p.a, p.b, p.c, p.gamma, p.d, p.alpha, p.h}
	ss := make([]string, len(vals))
	for i, x := range vals {
		ss[i] = strconv.FormatFloat(x, 'g', 17, 64)
	}
	b := []byte("FractionalChen4D-Key-v1|" + strings.Join(ss, "|") + "|")
	b = append(b, salt...)
	return sha256.Sum256(b)
}
func chenStateBytes(state [4]float64) []byte {
	parts := make([]string, 4)
	for i, v := range state {
		s := strconv.FormatFloat(v, 'f', 9, 64)
		if s == "-0.000000000" {
			s = "0.000000000"
		}
		parts[i] = s
	}
	return []byte("ChenState-CFP9-v1|" + strings.Join(parts, "|"))
}

// Fractional-order Adams-Bashforth-Moulton (exact protocol equations). No Python execution.
func chenStream(length int, original chenParams, salt []byte) ([]byte, error) {
	if length <= 0 || length > 1<<26 {
		return nil, fmt.Errorf("invalid Chen output length %d", length)
	}
	p := chenSalted(original, salt)
	const discard = 100
	count := (length + 31) / 32
	total := count + discard
	states := make([][4]float64, total)
	deriv := make([][4]float64, total)
	states[0] = p.initial
	deriv[0] = chenSystem(p, states[0])
	pa := math.Pow(p.h, p.alpha)
	predScale := pa / math.Gamma(p.alpha+1)
	corrScale := pa / math.Gamma(p.alpha+2)
	for n := 0; n < total-1; n++ {
		var sumP, sumC [4]float64
		for j := 0; j <= n; j++ {
			pf := math.Pow(float64(n+1-j), p.alpha) - math.Pow(float64(n-j), p.alpha)
			for k := 0; k < 4; k++ {
				sumP[k] += pf * deriv[j][k]
			}
		}
		var predicted [4]float64
		for k := 0; k < 4; k++ {
			predicted[k] = p.initial[k] + predScale*sumP[k]
		}
		fd := chenSystem(p, predicted)
		w0 := math.Pow(float64(n), p.alpha+1) - (float64(n)-p.alpha)*math.Pow(float64(n+1), p.alpha)
		for k := 0; k < 4; k++ {
			sumC[k] = w0 * deriv[0][k]
		}
		for j := 1; j <= n; j++ {
			dist := float64(n - j)
			cw := math.Pow(dist+2, p.alpha+1) + math.Pow(dist, p.alpha+1) - 2*math.Pow(dist+1, p.alpha+1)
			for k := 0; k < 4; k++ {
				sumC[k] += cw * deriv[j][k]
			}
		}
		var corrected [4]float64
		for k := 0; k < 4; k++ {
			corrected[k] = p.initial[k] + corrScale*(fd[k]+sumC[k])
			if math.IsNaN(corrected[k]) || math.IsInf(corrected[k], 0) || math.Abs(corrected[k]) > 1e12 {
				return nil, fmt.Errorf("divergent Chen state at step %d", n+1)
			}
		}
		states[n+1] = corrected
		deriv[n+1] = chenSystem(p, corrected)
	}
	keyContext := chenCtx(p, salt)
	out := make([]byte, 0, 32*count)
	for i := 0; i < count; i++ {
		st := chenStateBytes(states[discard+i])
		b := make([]byte, 0, len(keyContext)+8+len(st))
		b = append(b, keyContext[:]...)
		var cnt [8]byte
		binary.BigEndian.PutUint64(cnt[:], uint64(i))
		b = append(b, cnt[:]...)
		b = append(b, st...)
		hash := sha256.Sum256(b)
		out = append(out, hash[:]...)
	}
	return out[:length], nil
}
