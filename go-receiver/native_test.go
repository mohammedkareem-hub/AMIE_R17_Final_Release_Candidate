package main

import (
	"bytes"
	"os"
	"testing"
)

func TestIndependentGoFullNativeRecovery(t *testing.T) {
	pkg, key := vectorFixture(t)
	auth, err := verify(pkg, key)
	if err != nil {
		t.Fatal(err)
	}
	if !auth.OuterHMACVerified || !auth.InnerHMACVerified {
		t.Fatal("expected dual authentication")
	}
	dicom, err := decryptNative(pkg, key)
	if err != nil {
		t.Fatalf("Go native decryption: %v", err)
	}
	vec, err := parseVector(os.Getenv("AMIE_TEST_VECTOR"))
	if err != nil {
		t.Fatal(err)
	}
	expected := vec.ExpectedHash["original_dicom_sha256"]
	if len(expected) != 64 {
		t.Fatal("test vector does not declare original DICOM SHA256")
	}
	if got := shaHex(dicom); got != expected {
		t.Fatalf("Go recovered SHA256 %s expected %s", got, expected)
	}
	ctx := pkg.dicomContext
	prefix := pkg.fields["dicom-prefix"]
	suffix := pkg.fields["dicom-suffix"]
	if !bytes.Equal(dicom[:len(prefix)], prefix) || !bytes.Equal(dicom[len(dicom)-len(suffix):], suffix) {
		t.Fatal("DICOM envelope not restored")
	}
	pixelCount, _ := asInt(ctx["pixel_data_length_bytes"])
	if len(dicom) != len(prefix)+int(pixelCount)+len(suffix) {
		t.Fatal("wrong full DICOM size")
	}
	t.Logf("Go native DICOM byte-exact SHA-256 verified: %s, bytes %d", expected, len(dicom))
}
