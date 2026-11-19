package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

func vectorFixture(t *testing.T) (*packageData, []byte) {
	t.Helper()
	pkgFile := os.Getenv("AMIE_TEST_PACKAGE")
	vecFile := os.Getenv("AMIE_TEST_VECTOR")
	if pkgFile == "" || vecFile == "" {
		t.Skip("set AMIE_TEST_PACKAGE and AMIE_TEST_VECTOR to local research-only fixture paths")
	}
	raw, err := os.ReadFile(pkgFile)
	if err != nil {
		t.Fatal(err)
	}
	p, err := parsePackage(raw, defaultMaxPackageBytes)
	if err != nil {
		t.Fatal(err)
	}
	v, err := parseVector(vecFile)
	if err != nil {
		t.Fatal(err)
	}
	key, err := hex.DecodeString(v.Key)
	if err != nil {
		t.Fatal(err)
	}
	if shaHex(raw) != v.ExpectedHash["serialized_package_sha256"] {
		t.Fatal("wrong test vector or package")
	}
	return p, key
}
func clonePackage(p *packageData) *packageData {
	f := make(map[string][]byte)
	for k, v := range p.fields {
		f[k] = append([]byte{}, v...)
	}
	return &packageData{fields: f, metadata: p.metadata, dicomContext: p.dicomContext, raw: p.raw}
}
func TestGoldenVectorGoDualAuthentication(t *testing.T) {
	p, key := vectorFixture(t)
	r, err := verify(p, key)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Pass || !r.InnerHMACVerified || !r.OuterHMACVerified || r.NativeDICOMDecrypted {
		t.Fatalf("unexpected result: %+v", r)
	}
	if !bytes.Equal(roundTripBytes(p), p.raw) {
		t.Fatal("package serialization not stable")
	}
}
func TestModifiedPackageRejected(t *testing.T) {
	p, key := vectorFixture(t)
	for _, field := range labels {
		t.Run(field, func(t *testing.T) {
			pp := clonePackage(p)
			if len(pp.fields[field]) == 0 {
				pp.fields[field] = []byte{1}
			} else {
				pp.fields[field][len(pp.fields[field])-1] ^= 1
			}
			altered := roundTripBytes(pp)
			parsed, err := parsePackage(altered, defaultMaxPackageBytes)
			if err == nil {
				_, err = verify(parsed, key)
			}
			if err == nil {
				t.Fatal("altered field was accepted")
			}
		})
	}
}
func TestWrongKeyRejected(t *testing.T) {
	p, key := vectorFixture(t)
	key2 := append([]byte{}, key...)
	key2[0] ^= 1
	if _, err := verify(p, key2); err == nil {
		t.Fatal("wrong key accepted")
	}
}
func TestMalformedFramingRejected(t *testing.T) {
	p, _ := vectorFixture(t)
	cases := map[string][]byte{
		"bad-magic": append([]byte{}, p.raw...),
		"truncated": p.raw[:len(p.raw)-1],
		"trailing":  append(append([]byte{}, p.raw...), 0xff),
		"oversized": p.raw,
	}
	cases["bad-magic"][0] ^= 1
	for label, data := range cases {
		t.Run(label, func(t *testing.T) {
			cap := defaultMaxPackageBytes
			if label == "oversized" {
				cap = len(p.raw) - 1
			}
			if _, err := parsePackage(data, cap); err == nil {
				t.Fatal("malformed package accepted")
			}
		})
	}
}
func TestInnerMACIndependentAfterAuthorizedOuterReMAC(t *testing.T) {
	p, key := vectorFixture(t)
	pp := clonePackage(p)
	pp.fields["amie-inner-tag"][0] ^= 1
	nonce, err := hex.DecodeString(pp.metadata["session_nonce_hex"].(string))
	if err != nil {
		t.Fatal(err)
	}
	outKey, _ := derivedKey(key, nonce, dicomAuthContext)
	var msg bytes.Buffer
	msg.WriteString(dicomEnvelopeDomain)
	for _, l := range labels[:6] {
		msg.Write(frame(l, pp.fields[l]))
	}
	pp.fields["dicom-outer-tag"] = mac256(outKey, msg.Bytes())
	pp.raw = roundTripBytes(pp)
	r, err := verify(pp, key)
	if err == nil || !r.OuterHMACVerified || r.InnerHMACVerified {
		t.Fatalf("inner MAC not independently checked: %v, %+v", err, r)
	}
}
func TestPythonCanonicalJSON(t *testing.T) {
	expected := []byte(`{"a":"<>&","b":"\\u00e9","c":-1,"d":true,"e":null}`)
	var v any
	dec := json.NewDecoder(bytes.NewReader(expected))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		t.Fatal(err)
	}
	s, err := canonJSON(v)
	if err != nil {
		t.Fatal(err)
	}
	if s != string(expected) {
		t.Fatalf("unexpected canonical: %q", s)
	}
}
