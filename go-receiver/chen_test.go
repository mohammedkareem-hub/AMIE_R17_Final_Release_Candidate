package main

import (
	"encoding/binary"
	"encoding/hex"
	"strconv"
	"testing"
)

func TestChenAgainstFrozenReference(t *testing.T) {
	master, _ := hex.DecodeString("f04e8391f365be7de81d1dc0ee32848545b41abcd8b21d2bbd0d72235ff909c4")
	nonce, _ := hex.DecodeString("c4d870597a3d0939592d8dae57a5595c")
	p, err := chenDerived(master, nonce)
	if err != nil {
		t.Fatal(err)
	}
	salt, _ := derivedKey(master, nonce, "parallel-baker-salt")
	saltInt := binary.BigEndian.Uint64(salt)
	seed, err := chenStream(64, p, []byte("baker-diffusion|"+strconv.FormatUint(saltInt, 10)))
	if err != nil {
		t.Fatal(err)
	}
	want := "83f881754d42f0aa2a109d84acc01f9dd3f4b2f1a938b45fa8fef248f7b92523cb76304ed953ef471e58dc3c9f9dd11f026bd7d61f34b9523ad62b667f7969c9"
	if hex.EncodeToString(seed) != want {
		t.Fatalf("Go Chen64 got %x expected %s; derived initial %v", seed, want, p.initial)
	}
}
