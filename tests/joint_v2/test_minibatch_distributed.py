"""Distributed minibatch coverage, tail handling, and compressed record compatibility."""
import numpy as np
import pytest

from apexgen.joint_v2.runtime.minibatch_distributed import rank_indices, validate_batch_layout
from apexgen.joint_v2.runtime.portable import training_batches
from apexgen.shared.storage.store import RECORD_SCHEMA_VERSION, pack_record, unpack_record


def test_epoch_coverage_and_exact_resume_order():
    batches=list(training_batches(24576,96,42,0,256))
    validate_batch_layout(24576,96,3)
    seen=[]
    for _,indices in batches:
        shards=[rank_indices(indices,rank,3) for rank in range(3)]
        assert [len(s) for s in shards]==[32,32,32]
        flat=[i for shard in shards for i in shard]
        assert sorted(flat)==sorted(indices)
        seen.extend(flat)
    assert sorted(seen)==list(range(24576))
    assert list(training_batches(24576,96,42,127,256))==batches[127:]


def test_uneven_tail_keeps_every_sample_and_rejects_empty_rank():
    validate_batch_layout(19,12,3)
    batches=list(training_batches(19,12,7,0,2))
    assert [len(rank_indices(batches[-1][1],rank,3)) for rank in range(3)]==[3,2,2]
    assert sorted(i for _,batch in batches for i in batch)==list(range(19))
    with pytest.raises(ValueError,match='final minibatch'):
        validate_batch_layout(25,12,3)


@pytest.mark.parametrize('compression',[None,'zlib'])
def test_upstream_compressed_record_is_lossless(compression):
    record=dict(schema_version=RECORD_SCHEMA_VERSION,x=np.arange(12,dtype=np.float32).reshape(4,3),
                mask=np.array([True,False]),nested=dict(value='test'))
    result=unpack_record(pack_record(record,compression=compression))
    np.testing.assert_array_equal(result['x'],record['x'])
    np.testing.assert_array_equal(result['mask'],record['mask'])
    assert result['nested']==record['nested']
