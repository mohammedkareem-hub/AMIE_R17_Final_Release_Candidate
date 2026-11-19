package main
import (
  "encoding/hex"
  "testing"
)
func TestSHAKE256Known(t *testing.T) {
    // FIPS 202 empty message, SHAKE256 64 bytes
    want := "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"+"d75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
    if got:=hex.EncodeToString(shake256(nil,64));got!=want{t.Fatalf("SHAKE256 mismatch got %s",got)}
}
