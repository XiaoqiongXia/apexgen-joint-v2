"""Backward-compatible, lossless compressed LMDB record decoding."""
import numpy as np
import pytest
import zlib
from apexgen.shared.storage.store import RECORD_SCHEMA_VERSION, pack_record, unpack_record


def test_raw_and_compressed_records_decode_identically():
    record = dict(schema_version=RECORD_SCHEMA_VERSION, sample_id='example',
        coordinates=np.arange(120, dtype=np.float32).reshape(10, 4, 3),
        mask=np.ones((10, 4), dtype=bool), nested=dict(sequence='ACDEFGHIKL'))
    raw = pack_record(record)
    compressed = pack_record(record, compression='zlib')
    assert pack_record(unpack_record(raw)) == raw
    assert pack_record(unpack_record(compressed)) == raw
    assert len(compressed) < len(raw)
    with pytest.raises(zlib.error):
        unpack_record(compressed[:-3])
