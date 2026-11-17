# SecureDamage evaluation summary

Machine: x86_64, 10 CPU threads, Python 3.12.3, OpenFHE 1.6.1.0

## Non-private baseline (plaintext numpy)

- ReLU model: test accuracy 0.8550, latency 0.219 ms/image
- Polynomial model: test accuracy 0.8425, latency 0.227 ms/image
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
| Max abs logit error (worst image) | 7.93e-07 | 1.05e-09 | 5.99e-07 |
| Median abs logit error | 6.25e-08 | 8.35e-11 | 8.61e-08 |
| Smallest plaintext logit margin | 0.008 | 0.008 | 0.008 |
| Client encrypt (s) | 0.067 ± 0.022 | 0.118 ± 0.016 | 0.110 ± 0.003 |
| Server evaluate (s) | 0.440 ± 0.041 | 0.834 ± 0.074 | 0.796 ± 0.019 |
| Client decrypt (s) | 8.42 ms | 17.27 ms | 16.61 ms |
| Upload per query (MB) | 17.51 | 35.01 | 35.01 |
| Download per query (MB) | 0.50 | 1.00 | 1.00 |
| Rotation keys (MB, once) | 162.0 | 270.0 | 270.0 |
| Relinearization key (MB, once) | 9.0 | 15.0 | 15.0 |
| Public key (MB, once) | 1.75 | 3.50 | 3.50 |
| Client key generation (s, once) | 0.10 | 0.15 | 0.15 |
