package main

// Minimal SHAKE256 (FIPS 202) implementation for deterministic interoperability.
// Implements Keccak-f[1600] and the SHAKE256 136-byte rate. Uses no external modules.
import (
	"encoding/binary"
	"math/bits"
)

var keccakRC = [24]uint64{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
	0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
	0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
	0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
	0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
}

// Rho offset for lane x+5*y (x changes fastest); indexed by x,y.
var keccakRho = [25]int{
	0, 1, 62, 28, 27,
	36, 44, 6, 55, 20,
	3, 10, 43, 25, 39,
	41, 45, 15, 21, 8,
	18, 2, 61, 56, 14,
}

func keccakF(a *[25]uint64) {
	for _, rc := range keccakRC {
		var c [5]uint64
		for x := 0; x < 5; x++ {
			c[x] = a[x] ^ a[x+5] ^ a[x+10] ^ a[x+15] ^ a[x+20]
		}
		for x := 0; x < 5; x++ {
			d := c[(x+4)%5] ^ bits.RotateLeft64(c[(x+1)%5], 1)
			for y := 0; y < 5; y++ {
				a[x+5*y] ^= d
			}
		}
		var b [25]uint64
		for x := 0; x < 5; x++ {
			for y := 0; y < 5; y++ {
				b[y+5*((2*x+3*y)%5)] = bits.RotateLeft64(a[x+5*y], keccakRho[x+5*y])
			}
		}
		for x := 0; x < 5; x++ {
			for y := 0; y < 5; y++ {
				a[x+5*y] = b[x+5*y] ^ ((^b[(x+1)%5+5*y]) & b[(x+2)%5+5*y])
			}
		}
		a[0] ^= rc
	}
}

func shake256(input []byte, outputLen int) []byte {
	const rate = 136
	var state [25]uint64
	for len(input) >= rate {
		for j := 0; j < rate/8; j++ {
			state[j] ^= binary.LittleEndian.Uint64(input[j*8:])
		}
		keccakF(&state)
		input = input[rate:]
	}
	block := make([]byte, rate)
	copy(block, input)
	block[len(input)] = 0x1f
	block[rate-1] ^= 0x80
	for j := 0; j < rate/8; j++ {
		state[j] ^= binary.LittleEndian.Uint64(block[j*8:])
	}
	keccakF(&state)
	out := make([]byte, outputLen)
	for offset := 0; offset < outputLen; {
		var buf [rate]byte
		for j := 0; j < rate/8; j++ {
			binary.LittleEndian.PutUint64(buf[j*8:], state[j])
		}
		n := copy(out[offset:], buf[:])
		offset += n
		if offset < outputLen {
			keccakF(&state)
		}
	}
	return out
}
