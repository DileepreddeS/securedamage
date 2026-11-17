# SecureDamage evaluation summary

Machine: x86_64, 2 CPU threads, Python 3.12.13, OpenFHE 1.6.1.0

## Non-private baseline (plaintext numpy)

- ReLU model: test accuracy 0.8550, latency 0.270 ms/image
- Polynomial model: test accuracy 0.8425, latency 0.275 ms/image
- Query size: 3072 bytes (raw 32x32x3 uint8 image)

## Encrypted inference

| | Set A | Set B | Set C |
|---|---|---|---|
| Parameters | 128-bit, scale 2^40 | 128-bit, scale 2^50 | 192-bit, scale 2^40 |
| Ring dimension N | 16384 | 32768 | 32768 |
| log2 Q | 279 | 330 | 280 |
| OpenFHE at N/2 | rejected | rejected | rejected |
| Images evaluated | 1200 | 300 | 300 |
| Argmax agreement with plaintext | 100.00% | 100.00% | 100.00% |
| Encrypted accuracy | 84.25% | 82.00% | 82.00% |
| Max abs logit error (worst image) | 5.74e-07 | 7.34e-10 | 1.06e-06 |
| Median abs logit error | 5.88e-08 | 8.24e-11 | 8.78e-08 |
| Smallest plaintext logit margin | 0.008 | 0.008 | 0.008 |
| Client encrypt (s) | 0.104 ± 0.012 | 0.203 ± 0.023 | 0.200 ± 0.021 |
| Server evaluate (s) | 0.484 ± 0.040 | 1.038 ± 0.072 | 1.044 ± 0.064 |
| Client decrypt (s) | 11.46 ms | 26.00 ms | 26.08 ms |
| Upload per query (MB) | 17.51 | 35.01 | 35.01 |
| Download per query (MB) | 0.50 | 1.00 | 1.00 |
| Rotation keys (MB, once) | 162.0 | 270.0 | 270.0 |
| Relinearization key (MB, once) | 9.0 | 15.0 | 15.0 |
| Public key (MB, once) | 1.75 | 3.50 | 3.50 |
| Client key generation (s, once) | 0.36 | 0.49 | 0.60 |
