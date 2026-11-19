package main

// Security fuzz harness: CFP9-v2 native DICOM framing and metadata.
// Copy this file to the existing go-receiver/ directory. The tests do NOT
// alter the frozen public test vector or encryption/decryption algorithm.

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestPhase7BakerPartitionNoPanic(t *testing.T) {
	enc := make([]byte, 64*64)
	key := make([]byte, 64*64)
	invalid := [][]int{
		nil, {}, {64, 64}, {32, 32, 32}, {63, 1}, {0, 64}, {-1, 64}, {32}, {1, 1}, {65},
	}
	for _, partition := range invalid {
		// If inverseBakerBlock panics, go test fails; expected behavior is error.
		if _, err := inverseBakerBlock(enc, key, 64, partition); err == nil {
			t.Fatalf("invalid partition unexpectedly accepted: %v", partition)
		}
	}
	for _, valid := range [][]int{{64}, {32, 32}, {8, 8, 8, 8, 8, 8, 8, 8}} {
		if _, err := inverseBakerBlock(enc, key, 64, valid); err != nil {
			t.Fatalf("valid partition %v rejected: %v", valid, err)
		}
	}
}

func TestPhase7DeterministicMutationalParser(t *testing.T) {
	r := rand.New(rand.NewSource(20261104))
	capBytes := 1 << 20
	// Independent malformed binary inputs: no panic and no unbounded allocation.
	for i := 0; i < 25000; i++ {
		size := r.Intn(1024)
		data := make([]byte, size)
		if _, err := r.Read(data); err != nil {
			t.Fatal(err)
		}
		p, err := parsePackage(data, capBytes)
		if err == nil {
			if !bytes.Equal(roundTripBytes(p), data) {
				t.Fatal("parser accepted nonroundtripping bytes")
			}
		}
	}
	fixture, err := os.ReadFile(filepath.Join("..", "synthetic-public-vector", "synthetic_128x128.amiedcm"))
	if err != nil {
		t.Skip("synthetic test-vector fixture not available")
	}
	vectorJSON, err := os.ReadFile(filepath.Join("..", "synthetic-public-vector", "synthetic_128x128_test_vector.json"))
	if err != nil {
		t.Skip("synthetic test-vector metadata not available")
	}
	var vector struct {
		Key string `json:"validation_master_key_hex"`
	}
	if err := json.Unmarshal(vectorJSON, &vector); err != nil {
		t.Fatal(err)
	}
	master, err := hex.DecodeString(vector.Key)
	if err != nil || len(master) != 32 {
		t.Fatal("invalid public research test key")
	}
	for i := 0; i < 1000; i++ {
		copyB := bytes.Clone(fixture)
		at := r.Intn(len(copyB))
		copyB[at] ^= byte(1 << uint(r.Intn(8)))
		pkg, err := parsePackage(copyB, capBytes)
		if err != nil {
			continue
		}
		// Well-formed unauthenticated byte mutations must be rejected by MAC.
		if _, err = verify(pkg, master); err == nil {
			t.Fatal("mutated package accepted by HMAC")
		}
	}
}

func FuzzPackageParser(f *testing.F) {
	// During ordinary go test, these serve as seed regression tests.
	f.Add([]byte{})
	f.Add([]byte(magic))
	f.Add([]byte("AMIE-NATIVE-DICOM-PACKAGE\x00\x01\x00\x07"))
	if raw, err := os.ReadFile(filepath.Join("..", "synthetic-public-vector", "synthetic_128x128.amiedcm")); err == nil {
		f.Add(raw)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > 1<<20 {
			return
		}
		pkg, err := parsePackage(raw, 1<<20)
		if err != nil {
			return
		}
		if !bytes.Equal(roundTripBytes(pkg), raw) {
			t.Fatalf("parsePackage accepted nonroundtripping binary")
		}
		// Verify should fail closed on untrusted package bytes, no panic.
		_, _ = verify(pkg, make([]byte, 32))
	})
}

func FuzzCanonicalJSON(f *testing.F) {
	for _, s := range []string{"", "{}", "[]", "{\"x\":1}", "{\"x\":1,\"y\":[2]}", "{ \"x\" : 1}", "{\"x\":NaN}", "{\"a\":1,\"a\":2}", "{\"unicode\":\"\\u0645\"}"} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) > 1<<20 {
			return
		}
		obj, err := canonicalDict(raw)
		if err != nil {
			return
		}
		encoded, err := canonJSON(obj)
		if err != nil || string(raw) != encoded {
			t.Fatalf("accepted noncanonical JSON")
		}
	})
}
