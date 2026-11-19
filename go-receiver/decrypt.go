package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
)

// Global inverse diffusion: first reverse backward pass, then forward pass.
func inverseGlobal(ciphertext, forwardKey, backwardKey []byte) ([]byte, error) {
	n := len(ciphertext)
	if n == 0 || len(forwardKey) != n || len(backwardKey) != n {
		return nil, errors.New("global key length mismatch")
	}
	forward := make([]byte, n)
	forward[n-1] = ciphertext[n-1] - backwardKey[n-1]
	for i := n - 2; i >= 0; i-- {
		forward[i] = ciphertext[i] - backwardKey[i] - ciphertext[i+1]
	}
	plain := make([]byte, n)
	plain[0] = forward[0] - forwardKey[0]
	for i := 1; i < n; i++ {
		plain[i] = forward[i] - forwardKey[i] - forward[i-1]
	}
	return plain, nil
}
func expandSHA256(seed []byte, length int, domain string) []byte {
	out := make([]byte, 0, length+32)
	for counter := uint64(0); len(out) < length; counter++ {
		h := sha256.New()
		h.Write([]byte(domain))
		var c [8]byte
		binary.BigEndian.PutUint64(c[:], counter)
		h.Write(c[:])
		h.Write(seed)
		out = append(out, h.Sum(nil)...)
	}
	return out[:length]
}
func permutation(p chenParams, salt uint64, height, width int) ([]int, error) {
	if height <= 0 || width <= 0 || height*width > 1<<26 {
		return nil, errors.New("invalid permutation shape")
	}
	n := height * width
	sb := []byte(strconv.FormatUint(salt, 10))
	chaotic, err := chenStream(64, p, append([]byte("permutation-chaos|"), sb...))
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	h.Write([]byte("GlobalPermutationSeed-v2|"))
	h.Write(chaotic)
	h.Write([]byte("|"))
	var dimensions [16]byte
	binary.BigEndian.PutUint64(dimensions[:8], uint64(height))
	binary.BigEndian.PutUint64(dimensions[8:], uint64(width))
	h.Write(dimensions[:])
	h.Write([]byte("|"))
	h.Write(sb)
	seed := h.Sum(nil)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	rng := newPermutationShakeStream(seed)
	for i := n - 1; i > 0; i-- {
		j := rng.randBelow(i + 1)
		perm[i], perm[j] = perm[j], perm[i]
	}
	return perm, nil
}

type permShakeStream struct {
	seed    []byte
	counter uint64
	buffer  []byte
	offset  int
}

func newPermutationShakeStream(seed []byte) *permShakeStream { return &permShakeStream{seed: seed} }
func (s *permShakeStream) readByte() byte {
	if s.offset >= len(s.buffer) {
		data := make([]byte, 0, len("PermutationSHAKE256-v1|")+8+len(s.seed))
		data = append(data, []byte("PermutationSHAKE256-v1|")...)
		var c [8]byte
		binary.BigEndian.PutUint64(c[:], s.counter)
		s.counter++
		data = append(data, c[:]...)
		data = append(data, s.seed...)
		s.buffer = shake256(data, 64)
		s.offset = 0
	}
	b := s.buffer[s.offset]
	s.offset++
	return b
}
func (s *permShakeStream) randBelow(upper int) int {
	if upper <= 1 {
		return 0
	}
	bitLen := 0
	for x := upper; x > 0; x >>= 1 {
		bitLen++
	}
	count := (bitLen + 7) / 8
	space := uint64(1) << (8 * count)
	limit := space - space%uint64(upper)
	for {
		var v uint64
		for j := 0; j < count; j++ {
			v = v<<8 | uint64(s.readByte())
		}
		if v < limit {
			return int(v % uint64(upper))
		}
	}
}

func globalDiffusionKeys(p chenParams, salt uint64, context []byte, height, width int) ([]byte, []byte, error) {
	if len(context) != 32 {
		return nil, nil, errors.New("plaintext context must be 32 bytes")
	}
	sb := []byte(strconv.FormatUint(salt, 10))
	chSalt := append(append(append([]byte("global-diffusion-seed|"), sb...), []byte("|")...), context...)
	chaotic, err := chenStream(64, p, chSalt)
	if err != nil {
		return nil, nil, err
	}
	seed := make([]byte, 0, 64+16+1+len(sb)+1+32)
	seed = append(seed, chaotic...)
	var dim [16]byte
	binary.BigEndian.PutUint64(dim[:8], uint64(height))
	binary.BigEndian.PutUint64(dim[8:], uint64(width))
	seed = append(seed, dim[:]...)
	seed = append(seed, '|')
	seed = append(seed, sb...)
	seed = append(seed, '|')
	seed = append(seed, context...)
	n := height * width
	return expandSHA256(seed, n, "GlobalForwardDiffusion-v2|"), expandSHA256(seed, n, "GlobalBackwardDiffusion-v2|"), nil
}

func bakerKey(p chenParams, length int, salt uint64) ([]byte, error) {
	sb := []byte(strconv.FormatUint(salt, 10))
	seed, err := chenStream(64, p, []byte("baker-diffusion|"+string(sb)))
	if err != nil {
		return nil, err
	}
	shaker := make([]byte, 0, 64+len(sb)+len(seed))
	shaker = append(shaker, []byte("AMIE-BakerChen|BakerKeyExpand-v1|")...)
	var lengthBuf [8]byte
	binary.BigEndian.PutUint64(lengthBuf[:], uint64(length))
	shaker = append(shaker, lengthBuf[:]...)
	shaker = append(shaker, '|')
	shaker = append(shaker, sb...)
	shaker = append(shaker, '|')
	shaker = append(shaker, seed...)
	return shake256(shaker, length), nil
}
func inverseBakerBlock(enc, key []byte, blockSize int, partition []int) ([]byte, error) {
	n := blockSize * blockSize
	if len(enc) != n || len(key) != n {
		return nil, errors.New("invalid Baker block/key length")
	}
	// Phase 7 hardening: validate the complete partition BEFORE indexing.
	// A correctly HMAC-authenticated but malformed partition such as
	// [64, 64] for a 64x64 block must produce an error, never a panic.
	// This does not change the result for any valid Baker partition.
	if len(partition) == 0 {
		return nil, errors.New("empty Baker partition")
	}
	widthSum := 0
	for _, width := range partition {
		if width <= 0 || width > blockSize || blockSize%width != 0 || width > blockSize-widthSum {
			return nil, errors.New("Baker partition exceeds block width or has invalid strip")
		}
		widthSum += width
	}
	if widthSum != blockSize {
		return nil, errors.New("Baker partition does not cover block")
	}
	dec := make([]byte, n)
	dec[0] = enc[0] - key[0]
	for i := 1; i < n; i++ {
		dec[i] = enc[i] - key[i] - enc[i-1]
	}
	dest := make([]byte, n)
	pos := 0
	for _, width := range partition {
		if width <= 0 || blockSize%width != 0 {
			return nil, errors.New("invalid Baker partition")
		}
		scale := blockSize / width
		for r := pos; r < pos+width; r++ {
			for col := 0; col < blockSize; col++ {
				nr := scale*(r-pos) + col%scale
				nc := col/scale + pos
				dest[r*blockSize+col] = dec[nr*blockSize+nc]
			}
		}
		pos += width
	}
	if pos != blockSize {
		return nil, errors.New("Baker partition does not cover block")
	}
	return dest, nil
}
func inverseAllBaker(matrix []byte, h, w, blockSize int, partition []int, p chenParams, bakerSalt uint64) ([]byte, error) {
	if h%blockSize != 0 || w%blockSize != 0 {
		return nil, errors.New("Baker matrix dimensions not divisible by block size")
	}
	out := make([]byte, len(matrix))
	index := uint64(0)
	for y := 0; y < h; y += blockSize {
		for x := 0; x < w; x += blockSize {
			block := make([]byte, blockSize*blockSize)
			for r := 0; r < blockSize; r++ {
				copy(block[r*blockSize:(r+1)*blockSize], matrix[(y+r)*w+x:(y+r)*w+x+blockSize])
			}
			key, err := bakerKey(p, len(block), bakerSalt+index)
			if err != nil {
				return nil, fmt.Errorf("Baker key block %d: %w", index, err)
			}
			dec, err := inverseBakerBlock(block, key, blockSize, partition)
			if err != nil {
				return nil, err
			}
			for r := 0; r < blockSize; r++ {
				copy(out[(y+r)*w+x:(y+r)*w+x+blockSize], dec[r*blockSize:(r+1)*blockSize])
			}
			index++
		}
	}
	return out, nil
}
func metadataInt(m map[string]any, k string) (int, error) {
	a, e := asInt(m[k])
	if e != nil || a < 0 || a > 1<<28 {
		return 0, fmt.Errorf("invalid %s", k)
	}
	return int(a), nil
}
func metadataShape(m map[string]any, k string) (int, int, error) {
	s, ok := m[k].([]any)
	if !ok || len(s) != 2 {
		return 0, 0, fmt.Errorf("invalid %s shape", k)
	}
	a, e1 := asInt(s[0])
	b, e2 := asInt(s[1])
	if e1 != nil || e2 != nil || a <= 0 || b <= 0 || a*b > 1<<28 {
		return 0, 0, fmt.Errorf("invalid %s dimensions", k)
	}
	return int(a), int(b), nil
}
func salt64(master, nonce []byte, context string) (uint64, error) {
	v, e := derivedKey(master, nonce, context)
	if e != nil {
		return 0, e
	}
	return binary.BigEndian.Uint64(v[:8]), nil
}

// decryptNative independently reverses all AMIE processing and assembles original DICOM.
// Must be invoked only after package HMACs and authenticated context are verified.
func decryptNative(pkg *packageData, master []byte) ([]byte, error) {
	meta := pkg.metadata
	ctx := pkg.dicomContext
	ph, pw, err := metadataShape(meta, "padded_shape")
	if err != nil {
		return nil, err
	}
	oh, ow, err := metadataShape(meta, "original_shape")
	if err != nil {
		return nil, err
	}
	bs, err := metadataInt(meta, "block_size")
	if err != nil || bs <= 0 || bs > 1024 {
		return nil, errors.New("invalid block size")
	}
	if meta["dtype"] != "uint8" || ph%bs != 0 || pw%bs != 0 || oh > ph || ow > pw {
		return nil, errors.New("invalid AMIE image metadata")
	}
	if ph*pw != len(pkg.fields["amie-ciphertext"]) {
		return nil, errors.New("ciphertext length mismatch")
	}
	rows, _ := asInt(ctx["native_matrix_rows"])
	cols, _ := asInt(ctx["native_matrix_columns"])
	if int(rows) != oh || int(cols) != ow {
		return nil, errors.New("DICOM context / image size mismatch")
	}
	nonceHex, ok := meta["session_nonce_hex"].(string)
	if !ok {
		return nil, errors.New("missing nonce")
	}
	nonce, e := hex.DecodeString(nonceHex)
	if e != nil || len(nonce) != 16 {
		return nil, errors.New("invalid nonce")
	}
	contextHex, ok := meta["plaintext_context_hex"].(string)
	if !ok {
		return nil, errors.New("missing plaintext context")
	}
	pc, e := hex.DecodeString(contextHex)
	if e != nil || len(pc) != 32 {
		return nil, errors.New("invalid plaintext context")
	}
	param, e := chenDerived(master, nonce)
	if e != nil {
		return nil, e
	}
	permSalt, e := salt64(master, nonce, "global-permutation-salt")
	if e != nil {
		return nil, e
	}
	bakSalt, e := salt64(master, nonce, "parallel-baker-salt")
	if e != nil {
		return nil, e
	}
	globSalt, e := salt64(master, nonce, "global-diffusion-salt")
	if e != nil {
		return nil, e
	}
	fwd, back, e := globalDiffusionKeys(param, globSalt, pc, ph, pw)
	if e != nil {
		return nil, fmt.Errorf("global key generation: %w", e)
	}
	blockEncrypted, e := inverseGlobal(pkg.fields["amie-ciphertext"], fwd, back)
	if e != nil {
		return nil, e
	}
	partition := []int{}
	if meta["baker_partition"] == nil {
		if bs%8 == 0 {
			for i := 0; i < 8; i++ {
				partition = append(partition, bs/8)
			}
		} else {
			for i := 0; i < bs; i++ {
				partition = append(partition, 1)
			}
		}
	} else {
		pp, ok := meta["baker_partition"].([]any)
		if !ok {
			return nil, errors.New("invalid Baker partition")
		}
		for _, v := range pp {
			n, e := asInt(v)
			if e != nil || n <= 0 || n > int64(bs) {
				return nil, errors.New("invalid Baker partition")
			}
			partition = append(partition, int(n))
		}
	}
	permuted, e := inverseAllBaker(blockEncrypted, ph, pw, bs, partition, param, bakSalt)
	if e != nil {
		return nil, e
	}
	permutation, e := permutation(param, permSalt, ph, pw)
	if e != nil {
		return nil, fmt.Errorf("permutation generation: %w", e)
	}
	padded := make([]byte, len(permuted))
	for i, idx := range permutation {
		padded[idx] = permuted[i]
	}
	// Verify plaintext-sensitive context after successful reverse transformation.
	digest := sha256.New()
	digest.Write([]byte("MedicalImagePlaintext-v1|"))
	var dim [16]byte
	binary.BigEndian.PutUint64(dim[:8], uint64(ph))
	binary.BigEndian.PutUint64(dim[8:], uint64(pw))
	digest.Write(dim[:])
	digest.Write(padded)
	contextKey, e := derivedKey(master, nonce, "plaintext-context-key")
	if e != nil {
		return nil, e
	}
	expected := mac256(contextKey, append([]byte("PlaintextContext-v1|"), digest.Sum(nil)...))
	if !bytes.Equal(expected, pc) {
		return nil, errors.New("recovered plaintext context mismatch; inverse transform not interoperable")
	}
	// Crop padded uint8 matrix in the same row-major order as Python remove_padding.
	pixels := make([]byte, oh*ow)
	for r := 0; r < oh; r++ {
		copy(pixels[r*ow:(r+1)*ow], padded[r*pw:r*pw+ow])
	}
	payloadLen, e := asInt(ctx["pixel_data_length_bytes"])
	if e != nil || int(payloadLen) != len(pixels) {
		return nil, errors.New("recovered native Pixel Data length mismatch")
	}
	reconstructed := make([]byte, 0, len(pkg.fields["dicom-prefix"])+len(pixels)+len(pkg.fields["dicom-suffix"]))
	reconstructed = append(reconstructed, pkg.fields["dicom-prefix"]...)
	reconstructed = append(reconstructed, pixels...)
	reconstructed = append(reconstructed, pkg.fields["dicom-suffix"]...)
	return reconstructed, nil
}
