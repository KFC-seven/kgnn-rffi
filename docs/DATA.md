# Data format

The repository does not redistribute WiSig. The paper pipeline reads the compact pickle representation used by the WiSig preprocessing workflow.

Each pickle must contain a dictionary with these keys:

```python
{
    "tx_list": [...],
    "rx_list": [...],
    "capture_date_list": [...],
    "equalized_list": [0, 1],
    "data": nested_array
}
```

`data[tx_index][rx_index][date_index][equalized_index]` is a NumPy array with shape `(samples, length, 2)`. The last dimension stores the real-valued I and Q components. The paper uses equalized samples (`equalized=1`) and applies per-sample RMS power normalization before feature extraction.

The supplied configurations expect:

- source date: `2021_03_15`
- held-out capture date: `2021_03_01`
- 12 receiver identifiers
- 6 transmitters for ManySig
- the filtered 100-transmitter population for ManyTx

`dpr_rffi.data.splits.build_manifest` validates the requested transmitter, receiver, date, and sample-count constraints before constructing a protocol. Protocol seeds and transmitter repetitions are fixed in `configs/`.

Dataset paths may be overridden without changing the configurations:

```bash
python scripts/run_protocol.py ... --data /absolute/path/to/ManySig.pkl
```
