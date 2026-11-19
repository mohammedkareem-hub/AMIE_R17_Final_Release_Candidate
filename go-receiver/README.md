# Go CFP9-v2 independent native-DICOM receiver

This independent Go implementation verifies both HMAC layers and reconstructs
the exact synthetic DICOM from a CFP9-v2 package, requiring no Python runtime.
The published example uses an entirely fabricated DICOM vector.

From repository root:

```sh
cd go-receiver
go test ./...
go run . -package ../synthetic-public-vector/synthetic_128x128.amiedcm -vector ../synthetic-public-vector/synthetic_128x128_test_vector.json
```

For **non-skipped integration tests** set `AMIE_TEST_PACKAGE` and
`AMIE_TEST_VECTOR` to the synthetic package and vector JSON first.
Go source crypto code is copied from private commit `baa8a6e` without changes.
Tests are adjusted to use only synthetic fixtures.

**Scope:** experimental interoperability validation, not a formal security proof.
