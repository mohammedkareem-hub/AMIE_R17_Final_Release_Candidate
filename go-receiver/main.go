// AMIE-BakerChen independent Go interoperability verifier, phase 2.
// No Python runtime, Python modules, or external Go modules are used.
// Includes full inverse AMIE image transform and native DICOM recovery
// after verifying both HMAC layers. No Python runtime is used.
package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	magic                  = "AMIE-NATIVE-DICOM-PACKAGE\x00\x01"
	algorithmVersion       = "AMIE-BakerChen-CFP9-v2"
	dicomProfile           = "AMIE-Native-DICOM-CFP9-v1"
	hkdfInfoPrefix         = "MedicalImageEncryption-v1|" // actual frozen protocol label
	dicomAuthContext       = "native-dicom-envelope-authentication-key"
	innerAuthContext       = "ciphertext-authentication-key"
	dicomEnvelopeDomain    = "AMIE-Native-DICOM-Envelope-v1|"
	innerMessageDomain     = "Ciphertext-v1|"
	defaultMaxPackageBytes = 32 << 20
)

var labels = []string{
	"dicom-prefix", "dicom-suffix", "dicom-context", "amie-metadata",
	"amie-ciphertext", "amie-inner-tag", "dicom-outer-tag",
}

func shaHex(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func mac256(key, message []byte) []byte {
	h := hmac.New(sha256.New, key)
	_, _ = h.Write(message)
	return h.Sum(nil)
}

func derivedKey(masterKey, nonce []byte, context string) ([]byte, error) {
	if len(masterKey) != 32 || len(nonce) != 16 || context == "" {
		return nil, errors.New("invalid HKDF parameters")
	}
	prk := mac256(nonce, masterKey)                                // RFC 5869 HKDF-Extract
	return mac256(prk, []byte(hkdfInfoPrefix+context+"\x01")), nil // first HKDF-Expand block
}

type packageData struct {
	fields       map[string][]byte
	metadata     map[string]any
	dicomContext map[string]any
	raw          []byte
}

func readSized(raw []byte, pos *int, n uint64) ([]byte, error) {
	// Avoid overflow and no allocations from untrusted length fields.
	if n > uint64(len(raw)-*pos) {
		return nil, errors.New("truncated or oversized frame")
	}
	end := *pos + int(n)
	out := raw[*pos:end]
	*pos = end
	return out, nil
}

func readUint(raw []byte, pos *int, count int) (uint64, error) {
	v, err := readSized(raw, pos, uint64(count))
	if err != nil {
		return 0, err
	}
	if count == 2 {
		return uint64(binary.BigEndian.Uint16(v)), nil
	}
	return binary.BigEndian.Uint64(v), nil
}

// Encode strings like Python's json.dumps(ensure_ascii=True), including Unicode surrogate pairs.
func quotePyJSONString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString("\\\"")
		case '\\':
			b.WriteString("\\\\")
		case '\b':
			b.WriteString("\\b")
		case '\f':
			b.WriteString("\\f")
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		case '\t':
			b.WriteString("\\t")
		default:
			if r < 0x20 || r > 0x7f {
				if r <= 0xffff {
					fmt.Fprintf(&b, "\\u%04x", r)
				} else {
					rr := uint32(r) - 0x10000
					fmt.Fprintf(&b, "\\u%04x\\u%04x", 0xd800+(rr>>10), 0xdc00+(rr&1023))
				}
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

func canonJSON(v any) (string, error) {
	switch x := v.(type) {
	case nil:
		return "null", nil
	case bool:
		if x {
			return "true", nil
		}
		return "false", nil
	case string:
		return quotePyJSONString(x), nil
	case json.Number:
		n := string(x)
		// Frozen CFP9 metadata/context use integer fields only. Reject floating
		// point numbers rather than silently disagreeing with Python formatting.
		if strings.ContainsAny(n, ".eE") {
			return "", errors.New("floating-point JSON not supported by frozen CFP9 Go profile")
		}
		if _, err := strconv.ParseInt(n, 10, 64); err != nil {
			return "", fmt.Errorf("integer field: %w", err)
		}
		return n, nil
	case []any:
		ss := make([]string, len(x))
		for i, e := range x {
			s, err := canonJSON(e)
			if err != nil {
				return "", err
			}
			ss[i] = s
		}
		return "[" + strings.Join(ss, ",") + "]", nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		ss := make([]string, 0, len(keys))
		for _, k := range keys {
			s, err := canonJSON(x[k])
			if err != nil {
				return "", err
			}
			ss = append(ss, quotePyJSONString(k)+":"+s)
		}
		return "{" + strings.Join(ss, ",") + "}", nil
	default:
		return "", fmt.Errorf("unsupported JSON value: %T", v)
	}
}

func canonicalDict(raw []byte) (map[string]any, error) {
	if !utf8.Valid(raw) {
		return nil, errors.New("invalid UTF-8 JSON")
	}
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("trailing JSON content")
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, errors.New("JSON field must be object")
	}
	canon, err := canonJSON(obj)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(raw, []byte(canon)) {
		return nil, errors.New("noncanonical JSON encoding")
	}
	return obj, nil
}

func frame(label string, payload []byte) []byte {
	b := make([]byte, 2+len(label)+8+len(payload))
	binary.BigEndian.PutUint16(b[:2], uint16(len(label)))
	copy(b[2:], label)
	binary.BigEndian.PutUint64(b[2+len(label):], uint64(len(payload)))
	copy(b[2+len(label)+8:], payload)
	return b
}

func parsePackage(raw []byte, maxBytes int) (*packageData, error) {
	if len(raw) > maxBytes {
		return nil, errors.New("package exceeds configured resource cap")
	}
	if len(raw) < len(magic)+2 || string(raw[:len(magic)]) != magic {
		return nil, errors.New("invalid package magic/version")
	}
	pos := len(magic)
	count, err := readUint(raw, &pos, 2)
	if err != nil {
		return nil, err
	}
	if count != 7 {
		return nil, fmt.Errorf("expected 7 frames, got %d", count)
	}
	fields := make(map[string][]byte)
	for _, wanted := range labels {
		ln, err := readUint(raw, &pos, 2)
		if err != nil {
			return nil, err
		}
		if ln == 0 {
			return nil, errors.New("empty frame label")
		}
		label, err := readSized(raw, &pos, ln)
		if err != nil {
			return nil, err
		}
		if string(label) != wanted {
			return nil, fmt.Errorf("unexpected frame order; expected %s", wanted)
		}
		plen, err := readUint(raw, &pos, 8)
		if err != nil {
			return nil, err
		}
		payload, err := readSized(raw, &pos, plen)
		if err != nil {
			return nil, err
		}
		fields[wanted] = payload
	}
	if pos != len(raw) {
		return nil, errors.New("trailing bytes after 7th frame")
	}
	ctx, err := canonicalDict(fields["dicom-context"])
	if err != nil {
		return nil, fmt.Errorf("dicom context: %w", err)
	}
	meta, err := canonicalDict(fields["amie-metadata"])
	if err != nil {
		return nil, fmt.Errorf("amie metadata: %w", err)
	}
	if ctx["profile"] != dicomProfile {
		return nil, errors.New("unsupported DICOM profile")
	}
	if meta["algorithm_version"] != algorithmVersion {
		return nil, errors.New("unsupported algorithm version")
	}
	if len(fields["amie-inner-tag"]) != 32 || len(fields["dicom-outer-tag"]) != 32 {
		return nil, errors.New("invalid HMAC tag length")
	}
	return &packageData{fields: fields, metadata: meta, dicomContext: ctx, raw: raw}, nil
}

func asInt(v any) (int64, error) {
	x, ok := v.(json.Number)
	if !ok {
		return 0, errors.New("expected integer JSON value")
	}
	return strconv.ParseInt(string(x), 10, 64)
}

func contextChecks(p *packageData) error {
	// Only after outer authentication. These checks do not modify the DICOM header.
	c := p.dicomContext
	if c["pixel_data_policy"] != "encrypted-and-authenticated" || c["container_policy"] != "retained-byte-exact-and-authenticated" {
		return errors.New("unsupported container policies")
	}
	for _, item := range []struct{ key, which string }{{"dicom_prefix_length_bytes", "dicom-prefix"}, {"dicom_suffix_length_bytes", "dicom-suffix"}} {
		n, err := asInt(c[item.key])
		if err != nil || n < 0 || int64(len(p.fields[item.which])) != n {
			return errors.New("retained DICOM length mismatch")
		}
	}
	for _, item := range []struct{ key, which string }{{"dicom_prefix_sha256", "dicom-prefix"}, {"dicom_suffix_sha256", "dicom-suffix"}} {
		if c[item.key] != shaHex(p.fields[item.which]) {
			return errors.New("retained DICOM hash mismatch")
		}
	}
	rows, err := asInt(c["native_matrix_rows"])
	if err != nil || rows <= 0 {
		return errors.New("invalid matrix rows")
	}
	cols, err := asInt(c["native_matrix_columns"])
	if err != nil || cols <= 0 {
		return errors.New("invalid matrix columns")
	}
	rr, err := asInt(c["rows"])
	if err != nil || rr <= 0 || rows != rr {
		return errors.New("invalid DICOM rows")
	}
	cc, err := asInt(c["columns"])
	if err != nil || cc <= 0 {
		return errors.New("invalid DICOM columns")
	}
	frames, err := asInt(c["number_of_frames"])
	if err != nil || frames != 1 {
		return errors.New("only single-frame DICOM supported")
	}
	samples, err := asInt(c["samples_per_pixel"])
	if err != nil || samples != 1 {
		return errors.New("only monochrome DICOM supported")
	}
	bits, err := asInt(c["bits_allocated"])
	if err != nil || bits <= 0 || bits%8 != 0 {
		return errors.New("invalid BitsAllocated")
	}
	stored, err := asInt(c["bits_stored"])
	if err != nil || stored <= 0 || stored > bits {
		return errors.New("invalid BitsStored")
	}
	highbit, err := asInt(c["high_bit"])
	if err != nil || highbit != stored-1 {
		return errors.New("inconsistent HighBit")
	}
	pixelLen, err := asInt(c["pixel_data_length_bytes"])
	if err != nil || pixelLen != rr*cc*bits/8 {
		return errors.New("inconsistent DICOM pixel data length")
	}
	if cols != cc*bits/8 || cols*rows != pixelLen {
		return errors.New("inconsistent native byte matrix shape")
	}
	representation, err := asInt(c["pixel_representation"])
	if err != nil || (representation != 0 && representation != 1) {
		return errors.New("unsupported PixelRepresentation")
	}
	bytesPerSample, err := asInt(c["bytes_per_sample"])
	if err != nil || bytesPerSample != bits/8 {
		return errors.New("inconsistent BytesPerSample")
	}
	if c["native_matrix_dtype"] != "uint8" {
		return errors.New("unsupported matrix dtype")
	}
	if metaShape, ok := p.metadata["padded_shape"].([]any); ok && len(metaShape) == 2 {
		ph, e := asInt(metaShape[0])
		if e != nil || ph <= 0 {
			return errors.New("invalid padded height")
		}
		pw, e := asInt(metaShape[1])
		if e != nil || pw <= 0 || ph*pw != int64(len(p.fields["amie-ciphertext"])) {
			return errors.New("ciphertext shape mismatch")
		}
	} else {
		return errors.New("missing ciphertext shape")
	}
	return nil
}

type result struct {
	Pass                    bool    `json:"pass"`
	Stage                   string  `json:"stage"`
	SerializedPackageSHA256 string  `json:"serialized_package_sha256"`
	CiphertextSHA256        string  `json:"ciphertext_sha256"`
	InnerHMACVerified       bool    `json:"inner_hmac_verified"`
	OuterHMACVerified       bool    `json:"outer_hmac_verified"`
	CiphertextShape         []int64 `json:"ciphertext_shape"`
	NativeDICOMDecrypted    bool    `json:"native_dicom_decrypted"`
	Note                    string  `json:"note"`
	RecoveredDICOMSHA256    string  `json:"recovered_dicom_sha256,omitempty"`
}

func verify(p *packageData, key []byte) (result, error) {
	result := result{Stage: "go-cross-language-authentication", SerializedPackageSHA256: shaHex(p.raw), CiphertextSHA256: shaHex(p.fields["amie-ciphertext"]), Note: "Go parser and independent dual HMAC verified. Use default full receiver for native DICOM recovery."}
	if !bytes.Equal(roundTripBytes(p), p.raw) {
		return result, errors.New("package serialization round-trip mismatch")
	}
	nonceHex, ok := p.metadata["session_nonce_hex"].(string)
	if !ok {
		return result, errors.New("missing nonce")
	}
	nonce, err := hex.DecodeString(nonceHex)
	if err != nil || len(nonce) != 16 {
		return result, errors.New("invalid nonce")
	}
	outerKey, err := derivedKey(key, nonce, dicomAuthContext)
	if err != nil {
		return result, err
	}
	innerKey, err := derivedKey(key, nonce, innerAuthContext)
	if err != nil {
		return result, err
	}
	var msg bytes.Buffer
	msg.WriteString(dicomEnvelopeDomain)
	for _, label := range labels[:6] {
		msg.Write(frame(label, p.fields[label]))
	}
	wantOuter := mac256(outerKey, msg.Bytes())
	if !hmac.Equal(wantOuter, p.fields["dicom-outer-tag"]) {
		return result, errors.New("outer HMAC authentication failed")
	}
	result.OuterHMACVerified = true
	if err := contextChecks(p); err != nil {
		return result, err
	}
	var inner bytes.Buffer
	inner.WriteString(innerMessageDomain)
	inner.Write(p.fields["amie-metadata"])
	inner.WriteByte('|')
	inner.Write(p.fields["amie-ciphertext"])
	wantInner := mac256(innerKey, inner.Bytes())
	if !hmac.Equal(wantInner, p.fields["amie-inner-tag"]) {
		return result, errors.New("inner HMAC authentication failed")
	}
	result.InnerHMACVerified = true
	shape := p.metadata["padded_shape"].([]any)
	r, _ := asInt(shape[0])
	c, _ := asInt(shape[1])
	result.CiphertextShape = []int64{r, c}
	result.Pass = true
	return result, nil
}

type testVector struct {
	Key          string            `json:"validation_master_key_hex"`
	ExpectedHash map[string]string `json:"expected_hashes"`
	ExpectedTags map[string]string `json:"expected_tags"`
}

func parseVector(filename string) (testVector, error) {
	var v testVector
	b, err := os.ReadFile(filename)
	if err != nil {
		return v, err
	}
	if err = json.Unmarshal(b, &v); err != nil {
		return v, err
	}
	return v, nil
}

func main() {
	pkgPath := flag.String("package", "", "path to .amiedcm binary package")
	vecPath := flag.String("vector", "", "path to public deterministic CFP9-v2 test vector (research only)")
	keyHex := flag.String("key-hex", "", "master key as hex; research samples are PUBLIC test keys")
	maxBytes := flag.Int("max-bytes", defaultMaxPackageBytes, "maximum accepted .amiedcm size")
	authOnly := flag.Bool("auth-only", false, "verify package and HMAC layers without decrypting native Pixel Data")
	outDICOM := flag.String("out-dicom", "", "write recovered DICOM to specified file (may contain patient information)")
	expectedHash := flag.String("expected-sha256", "", "expected original DICOM SHA-256 (optional; vector includes it)")
	flag.Parse()
	fail := func(err error) { fmt.Fprintln(os.Stderr, "FAIL:", err); os.Exit(1) }
	if *pkgPath == "" {
		fail(errors.New("-package is required"))
	}
	if *maxBytes <= 0 || *maxBytes > 2<<30 {
		fail(errors.New("invalid max-bytes"))
	}
	f, err := os.Open(*pkgPath)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		fail(err)
	}
	if info.Size() > int64(*maxBytes) {
		fail(errors.New("package file exceeds cap"))
	}
	raw, err := io.ReadAll(io.LimitReader(f, int64(*maxBytes)+1))
	if err != nil {
		fail(err)
	}
	if len(raw) > *maxBytes {
		fail(errors.New("package read exceeds cap"))
	}
	p, err := parsePackage(raw, *maxBytes)
	if err != nil {
		fail(err)
	}
	var vec testVector
	if *vecPath != "" {
		vec, err = parseVector(*vecPath)
		if err != nil {
			fail(err)
		}
		if *keyHex == "" {
			*keyHex = vec.Key
		}
		if expected := vec.ExpectedHash["serialized_package_sha256"]; expected != "" && expected != shaHex(raw) {
			fail(errors.New("serialized package SHA-256 differs from test vector"))
		}
		for ref, label := range map[string]string{"amie_ciphertext_sha256": "amie-ciphertext", "amie_metadata_canonical_json_sha256": "amie-metadata", "dicom_context_canonical_json_sha256": "dicom-context", "dicom_prefix_sha256": "dicom-prefix", "dicom_suffix_sha256": "dicom-suffix"} {
			if val := vec.ExpectedHash[ref]; val != "" && val != shaHex(p.fields[label]) {
				fail(fmt.Errorf("test vector hash mismatch: %s", ref))
			}
		}
		for ref, label := range map[string]string{"amie_inner_hmac_sha256_hex": "amie-inner-tag", "dicom_outer_hmac_sha256_hex": "dicom-outer-tag"} {
			if val := vec.ExpectedTags[ref]; val != "" && val != hex.EncodeToString(p.fields[label]) {
				fail(fmt.Errorf("test vector tag mismatch: %s", ref))
			}
		}
	}
	key, err := hex.DecodeString(*keyHex)
	if err != nil || len(key) != 32 {
		fail(errors.New("valid 256-bit -key-hex or -vector public validation key required"))
	}
	res, err := verify(p, key)
	if err != nil {
		fail(err)
	}
	if !*authOnly {
		recovered, err := decryptNative(p, key)
		if err != nil {
			fail(err)
		}
		digest := shaHex(recovered)
		expected := *expectedHash
		if expected == "" && *vecPath != "" {
			expected = vec.ExpectedHash["original_dicom_sha256"]
		}
		if expected != "" && !strings.EqualFold(expected, digest) {
			fail(fmt.Errorf("recovered DICOM SHA-256 mismatch: got %s expected %s", digest, expected))
		}
		res.RecoveredDICOMSHA256 = digest
		res.NativeDICOMDecrypted = true
		res.Stage = "go-cross-language-native-dicom-recovery"
		res.Note = "Independent Go receiver: outer HMAC, inner HMAC, Chen/CFP9, SHAKE256 permutation, inverse Baker and bidirectional diffusion, DICOM byte reconstruction verified."
		if *outDICOM != "" {
			if err := os.WriteFile(*outDICOM, recovered, 0600); err != nil {
				fail(err)
			}
		}
	}
	out, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		fail(err)
	}
	fmt.Println(string(out))
}

// roundTripBytes is used only as a protocol framing consistency check.
// The result is NOT a recovered medical image; ciphertext remains encrypted.
func roundTripBytes(p *packageData) []byte {
	var b bytes.Buffer
	b.WriteString(magic)
	count := make([]byte, 2)
	binary.BigEndian.PutUint16(count, 7)
	b.Write(count)
	for _, label := range labels {
		b.Write(frame(label, p.fields[label]))
	}
	return b.Bytes()
}
