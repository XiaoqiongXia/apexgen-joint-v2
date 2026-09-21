"""Published string-name arrays, row identities, and the model mapping boundary."""

from copy import deepcopy
import csv
import json
import weakref

import numpy as np
import pytest

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.boltz_npz import adapt_boltz_npz, audit_boltz_record, decode_atom_name
from apexgen.joint_v2.data.boltz_interfaces import describe_interfaces
from apexgen.joint_v2.data.boltz_chain_pairs import adapt_interface_direction
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.static_features import STATIC_FEATURES_SCHEMA
from test_boltz_npz import source_arrays, adapt
from test_boltz_chain_pairs import doubled_target, make_pair
from test_boltz_interfaces import fixture_arrays, serialized


def native_arrays(arrays):
    result = deepcopy(arrays)
    result['atoms'] = np.array([(decode_atom_name(a['name']), a['coords'], a['is_present'], 0., 1.)
        for a in arrays['atoms']], dtype=[('name', 'U4'), ('coords', 'f4', (3,)),
        ('is_present', '?'), ('bfactor', 'f4'), ('plddt', 'f4')])
    bonds = []
    for table in (arrays['bonds'], arrays['connections']):
        for bond in table:
            endpoints = []
            for side in (1, 2):
                ai = int(bond[f'atom_{side}'])
                ri = next(i for i,r in enumerate(arrays['residues'])
                    if r['atom_idx'] <= ai < r['atom_idx'] + r['atom_num'])
                ci = next(i for i,c in enumerate(arrays['chains'])
                    if c['res_idx'] <= ri < c['res_idx'] + c['res_num'])
                endpoints.append((ci, ri, ai))
            a,b = endpoints
            bonds.append((a[0],b[0],a[1],b[1],a[2],b[2],1))
    result['bonds'] = np.array(bonds, dtype=[(f'{name}_{side}', 'i4')
        for name in ('chain','res','atom') for side in (1,2)] + [('type','i1')])
    del result['connections']
    return result


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('direction', ['a_to_b', 'b_to_a'])
@pytest.mark.parametrize('omitted_count', [1, 11])
def test_fragment_contacts_survive_context_backbone_filtering(
    tmp_path, native, direction, omitted_count,
):
    path, original_pair = make_pair(tmp_path, model_length=12)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if native:
        arrays = native_arrays(arrays)
    context_start = 0 if direction == 'a_to_b' else 24
    for residue in arrays['residues'][context_start:context_start + omitted_count]:
        arrays['atoms']['is_present'][int(residue['atom_idx'])] = False
    np.savez(path, **arrays)
    source = {key: original_pair[key] for key in (
        'source_archive', 'source_member', 'source_npz_filename',
        'source_structure_id', 'source_offset',
    )}
    source['source_size'] = path.stat().st_size
    pairs, _ = describe_interfaces(path.read_bytes(), source)
    if omitted_count == 11:
        with pytest.raises(ValueError, match='target residues lost contact with retained context'):
            adapt_interface_direction(path, pairs[0], direction)
    else:
        record, _ = adapt_interface_direction(path, pairs[0], direction)
        assert record['peptide_length'] == 12
        assert len(record['pocket_aatype']) == 11
        assert record['interface_pair']['interface_density'] == 1.0


@pytest.mark.parametrize('aa', range(20))
def test_native_and_legacy_have_identical_model_identities(tmp_path, aa):
    arrays = source_arrays(aa)
    legacy = adapt(tmp_path, arrays)
    native = adapt(tmp_path, native_arrays(arrays))
    assert native['boltz_adapter']['source_schema'] == 'boltzgen_string_names'
    for field in ('aatype', 'experimental_atom14', 'experimental_atom14_mask'):
        np.testing.assert_array_equal(native['joint_v2_target'][field], legacy['joint_v2_target'][field])
    for field in ('pocket_aatype', 'pocket_atom_xyz', 'pocket_atom_mask'):
        np.testing.assert_array_equal(native[field], legacy[field])
    audit_boltz_record(native)
    # One-residue fixtures isolate all 20 vocab entries; model execution uses
    # continuous fragments of length >=4 in the end-to-end test below.
    batch = collate_joint_v2_records([native])
    assert batch.targets.endpoint_aatype[batch.condition.peptide_mask].tolist() == [aa]


def test_native_inventory_preserves_contacts_and_excludes_hydrogen():
    arrays = fixture_arrays()
    expected, _ = describe_interfaces(*serialized(arrays))
    names = ['C', 'N', 'C', 'C', 'C', 'O', 'C', 'C', 'H1']
    arrays['atoms'] = np.array([(n,a['coords'],a['is_present']) for n,a in zip(names,arrays['atoms'])],
        dtype=[('name','U4'),('coords','f4',(3,)),('is_present','?')])
    actual, _ = describe_interfaces(*serialized(arrays))
    for a,b in zip(actual,expected):
        for key in ('contact_verified','chain_a_interface_residue_rows','chain_b_interface_residue_rows',
                    'chain_a_observed_heavy_atom_count','chain_b_observed_heavy_atom_count'):
            assert a[key] == b[key]


@pytest.mark.parametrize('native', [False, True])
def test_connections_use_chain_rows_not_asym_ids(tmp_path, native):
    arrays = native_arrays(source_arrays()) if native else source_arrays()
    # A context-only connection remains legal; asym IDs intentionally differ.
    key = 'bonds' if native else 'connections'
    row = (0,0,0,0,0,1,1) if native else (0,0,0,0,0,1)
    arrays[key] = np.array([row],dtype=arrays[key].dtype)
    audit_boltz_record(adapt(tmp_path, arrays))
    arrays[key][0]['chain_1'] = 10
    with pytest.raises(ValueError, match='connection residue/chain identity'):
        adapt(tmp_path, arrays)
    arrays[key][0]['chain_1'] = 1
    with pytest.raises(ValueError, match='identities disagree'):
        adapt(tmp_path, arrays)


def test_native_absent_sidechain_keeps_mask_and_external_crosslink_rejects(tmp_path):
    arrays = native_arrays(source_arrays(1))
    start = int(arrays['residues'][-1]['atom_idx'])
    absent = next(i for i in range(start,len(arrays['atoms'])) if arrays['atoms'][i]['name']=='NH2')
    arrays['atoms'][absent]['is_present'] = False
    arrays['atoms'][absent]['coords'] = np.nan
    record = adapt(tmp_path,arrays)
    indices = record['boltz_adapter']['peptide_atom_source_indices']
    assert absent not in indices
    assert np.array_equal(indices >= 0,record['joint_v2_target']['experimental_atom14_mask'])
    audit_boltz_record(record)
    ni = next(i for i in range(start,len(arrays['atoms'])) if arrays['atoms'][i]['name']=='N')
    arrays['bonds'] = np.array([(2,0,2,0,ni,0,1)],dtype=arrays['bonds'].dtype)
    with pytest.raises(ValueError,match='crosslink'):
        adapt(tmp_path,arrays)


def test_native_fragment_boundary_bond_and_crosslink(tmp_path):
    arrays = native_arrays(doubled_target())
    def atom(ri,name):
        r=arrays['residues'][ri]
        return next(i for i in range(r['atom_idx'],r['atom_idx']+r['atom_num']) if arrays['atoms'][i]['name']==name)
    left,right=atom(2,'C'),atom(3,'N')
    arrays['bonds'] = np.array([(2,2,2,3,left,right,1)],dtype=arrays['bonds'].dtype)
    path=tmp_path/'fragment.npz'
    np.savez(path,**arrays)
    kwargs=dict(sample_id='fragment',source_pdb_id='synthetic',receptor_chain_ids=['A1'],
                peptide_chain_id='P1',target_residue_range=(1,2))
    record=adapt_boltz_npz(path,**kwargs)
    assert record['boltz_adapter']['cut_boundary_peptide_bonds']==[[left,right]]
    arrays['bonds'][0]['atom_2']=atom(3,'CA')
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='crosslink'):
        adapt_boltz_npz(path,**kwargs)


@pytest.mark.parametrize('change', ['name','res_type','observed_nan','missing_backbone','bond_shape','cyclic'])
def test_native_corruption_rejected(tmp_path,change):
    arrays=native_arrays(source_arrays())
    start=int(arrays['residues'][-1]['atom_idx'])
    ni=next(i for i in range(start,len(arrays['atoms'])) if arrays['atoms'][i]['name']=='N')
    if change=='name': arrays['atoms'][ni]['name']='CA'
    elif change=='res_type': arrays['residues'][-1]['res_type']=0
    elif change=='observed_nan': arrays['atoms'][ni]['coords']=np.nan
    elif change=='missing_backbone': arrays['atoms'][ni]['is_present']=False
    elif change=='bond_shape': arrays['bonds']=np.empty(0,dtype=[('atom_1','i4'),('atom_2','i4'),('type','i1')])
    else:
        chains=np.zeros(len(arrays['chains']),dtype=arrays['chains'].dtype.descr+[('cyclic_period','i4')])
        for name in arrays['chains'].dtype.names: chains[name]=arrays['chains'][name]
        chains['cyclic_period']=-1
        chains[-1]['cyclic_period']=1
        arrays['chains']=chains
    with pytest.raises(ValueError): adapt(tmp_path,arrays)


@pytest.mark.parametrize('field', ['mapping_sha256','schema','joint_v2_contract_sha256'])
def test_dataset_and_collate_reject_stale_mapping(tmp_path,field):
    record=adapt(tmp_path,native_arrays(source_arrays()))
    if field=='joint_v2_contract_sha256': record['joint_v2_target'][field]='0'*64
    else: record['boltz_adapter'][field]='0'*64
    with pytest.raises(ValueError,match='contract mismatch'):
        collate_joint_v2_records([record])
    dataset=JointV2Dataset.__new__(JointV2Dataset)
    dataset.pockets=[record]
    dataset.target_root=None
    dataset._static_features_schema=STATIC_FEATURES_SCHEMA
    with pytest.raises(ValueError,match='contract mismatch'):
        dataset[0]


def test_selected_native_files_reach_unified_dataset(tmp_path):
    from scripts.data.prepare_joint_v2_boltzgen import prepare_selected
    path,_=make_pair(tmp_path,model_length=9)
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    np.savez(path,**native_arrays(arrays))
    result=prepare_selected(tmp_path,['synthetic'],tmp_path/'output')
    dataset=JointV2Dataset(result['dataset'],split='smoke')
    try:
        records=[dataset[i] for i in range(len(dataset))]
        forward=[r for r in records if r['interface_pair']['direction']=='a_to_b']
        assert [r['boltz_adapter']['target_residue_range'] for r in forward]==[[0,4],[5,9]]
        for record in records: audit_boltz_record(record)
        collate_joint_v2_records(records).condition.validate_model_input()
        with (tmp_path/'output'/'sample_inventory.csv').open() as stream:
            inventory = list(csv.DictReader(stream))
        assert [r['sample_id'] for r in inventory] == [r['sample_id'] for r in records]
        selected = [r for r in inventory if r['direction'] == 'a_to_b']
        assert [(r['target_start_0based'], r['target_stop_0based_exclusive'])
                for r in selected] == [('0','4'), ('5','9')]
        assert [(r['target_start_1based'], r['target_end_1based_inclusive'])
                for r in selected] == [('1','4'), ('6','9')]
        for row in selected:
            assert row['source_archive'] == str(path.resolve())
            assert (row['condition_chain_id'], row['target_chain_id']) == ('A1','P1')
            assert row['target_source_length'] == row['condition_source_length'] == '9'
            assert row['target_length'] == '4' and row['target_sequence'] == 'GGGG'
        for row, record in zip(inventory, records):
            expanded = [i for lo,hi in json.loads(row['pocket_source_ranges_0based_half_open'])
                        for i in range(lo,hi)]
            assert expanded == [k['polymer_index'] for k in record['pocket_residue_keys']]
        assert result['sample_inventory']['samples'] == len(records)
        assert 'not PDB author residue numbers' in (tmp_path/'output'/'sample_inventory.md').read_text()
    finally: dataset.close()
    with pytest.raises(ValueError): prepare_selected(tmp_path,[],tmp_path/'empty')
    with pytest.raises(FileExistsError): prepare_selected(tmp_path,['synthetic'],tmp_path/'output')


def test_inventory_keeps_discontinuous_pocket_ranges():
    from scripts.data.prepare_joint_v2_boltzgen import _source_ranges
    assert _source_ranges([0,1,5,8,9]) == [[0,2],[5,6],[8,10]]


@pytest.mark.parametrize('with_valid_source', [False, True])
def test_native_stored_interface_without_observed_contact_is_logged(tmp_path, with_valid_source):
    from scripts.data.prepare_joint_v2_boltzgen import prepare_selected
    path, _ = make_pair(tmp_path, model_length=True)
    with np.load(path, allow_pickle=False) as archive:
        arrays = native_arrays({key: archive[key] for key in archive.files})
    np.savez(path, **arrays)
    target_start = int(arrays['chains'][-1]['atom_idx'])
    arrays['atoms']['coords'][target_start:] += 1000
    np.savez(tmp_path/'far.npz', **arrays)
    ids = ['far', 'synthetic'] if with_valid_source else ['far']
    result = prepare_selected(tmp_path, ids, tmp_path/'output')
    assert result['dataset_summary']['counts']['skipped_no_contact'] == 2
    expected = 2 if with_valid_source else 0
    assert result['dataset_summary']['counts']['included'] == expected
    with (tmp_path/'output'/'sample_inventory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    assert all(row['source_structure_id'] == 'synthetic' for row in rows)
    audits = [json.loads(line) for line in
              (tmp_path/'output'/'dataset'/'directions.jsonl').read_text().splitlines()]
    skipped = [row for row in audits if row['source_structure_id'] == 'far']
    assert len(skipped) == 2
    assert all(row['status'] == 'skipped_no_contact' and row['reason'] for row in skipped)


def test_sample_inventory_releases_atom_tensors_and_checks_source_metadata(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.data.prepare_joint_v2_boltzgen import prepare_selected, write_sample_inventory
    path, _ = make_pair(tmp_path, model_length=9)
    with np.load(path, allow_pickle=False) as archive:
        np.savez(tmp_path/'native.npz', **native_arrays({k: archive[k] for k in archive.files}))
    result = prepare_selected(tmp_path, ['native'], tmp_path/'output')
    assert result['sample_inventory']['samples'] >= 3
    refs = []
    original = JointV2Dataset.__getitem__

    def observe_lifetime(self, index):
        record = original(self, index)
        refs.append(weakref.ref(record['pocket_atom_xyz']))
        # CSV export must not retain earlier structures' large coordinate arrays.
        assert sum(ref() is not None for ref in refs) <= 2
        return record

    monkeypatch.setattr(JointV2Dataset, '__getitem__', observe_lifetime)
    inventory = tmp_path/'output'/'interfaces.parquet'
    exported = write_sample_inventory(result['dataset'], inventory, tmp_path/'report')
    assert exported['samples'] == result['sample_inventory']['samples']
    assert all(ref() is None for ref in refs)
    assert (tmp_path/'report'/'sample_inventory.csv').read_bytes() == (
        tmp_path/'output'/'sample_inventory.csv').read_bytes()
    table = pq.read_table(inventory)
    rows = table.to_pylist()
    rows[0]['chain_b_length'] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tmp_path/'wrong.parquet')
    with pytest.raises(ValueError, match='sample inventory/source identity mismatch'):
        write_sample_inventory(result['dataset'], tmp_path/'wrong.parquet', tmp_path/'badreport')
    assert not (tmp_path/'badreport'/'sample_inventory.csv').exists()


def test_new_build_does_not_reopen_lmdb_to_export_csv(tmp_path, monkeypatch):
    from scripts.data import prepare_joint_v2_boltzgen as preparation
    path, _ = make_pair(tmp_path, model_length=True)
    with np.load(path, allow_pickle=False) as archive:
        np.savez(tmp_path/'native.npz', **native_arrays({k: archive[k] for k in archive.files}))

    def unexpected_reread(*args, **kwargs):
        raise AssertionError('new builds must write CSV directly while processing records')

    monkeypatch.setattr(preparation, 'JointV2Dataset', unexpected_reread)
    monkeypatch.setattr(preparation, 'write_sample_inventory', unexpected_reread)
    result = preparation.prepare_selected(tmp_path, ['native'], tmp_path/'output')
    assert result['sample_inventory']['samples'] == 2
    assert (tmp_path/'output/sample_inventory.csv').read_bytes() == (
        tmp_path/'output/dataset/sample_inventory.csv').read_bytes()
    assert not (tmp_path/'output/dataset.inprogress').exists()
    assert json.loads((tmp_path/'output/dataset/build_status.json').read_text())['status'] == 'complete'


def test_cached_native_source_reuses_parse_and_preserves_every_field(tmp_path, monkeypatch):
    from apexgen.joint_v2.data.boltz_source import BoltzSource
    from apexgen.joint_v2.data.boltz_chain_pairs import adapt_interface_direction, fragment_selections
    from apexgen.shared.storage.store import pack_record
    path, _ = make_pair(tmp_path, model_length=9)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.savez(path, **native_arrays(arrays))
    payload = path.read_bytes()
    pairs, _ = describe_interfaces(payload, dict(source_archive=str(path), source_member=path.name,
        source_npz_filename=path.name, source_structure_id=path.stem, source_offset=0, source_size=len(payload)))
    pair = pairs[0]
    expected = []
    for direction in ('a_to_b', 'b_to_a'):
        for selection in fragment_selections(pair, direction):
            if selection['fragment_length'] >= 4:
                record, audit = adapt_interface_direction(path, pair, direction, fragment_index=selection['fragment_index'])
                expected.append((direction, selection['fragment_index'], pack_record(record), audit))
    cached = BoltzSource(path)
    def no_reopen(*args, **kwargs):
        raise AssertionError('cached conversion must not reopen the NPZ')
    monkeypatch.setattr(np, 'load', no_reopen)
    for _ in range(2):
        for direction, index, packed, audit in expected:
            actual, checked = adapt_interface_direction(path, pair, direction, fragment_index=index, source=cached)
            assert pack_record(actual) == packed
            assert checked == audit
    with pytest.raises(ValueError):
        cached.data['atoms'][0] = cached.data['atoms'][0]
    path.write_bytes(payload + b'changed')
    with pytest.raises(ValueError, match='changed after cached parsing'):
        cached.check_path(path)


def test_parallel_and_serial_builds_have_identical_samples_and_audits(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.data.prepare_joint_v2_boltz_chain_pairs import build_dataset
    from apexgen.joint_v2.data.boltz_interfaces import PAIR_SCHEMA
    from apexgen.shared.storage.store import pack_record
    _, pair = make_pair(tmp_path, model_length=9)
    inventory = tmp_path/'pairs.parquet'
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    serial, parallel = tmp_path/'serial', tmp_path/'parallel'
    a = build_dataset(inventory, serial, workers=1, shard_size=2)
    b = build_dataset(inventory, parallel, workers=2, shard_size=2, commit_size=64, compression='zlib')
    assert a['counts'] == b['counts']
    assert a['rejection_reasons'] == b['rejection_reasons']
    assert (serial/'directions.jsonl').read_bytes() == (parallel/'directions.jsonl').read_bytes()
    left, right = JointV2Dataset(serial, split='smoke'), JointV2Dataset(parallel, split='smoke')
    try:
        assert len(left) == len(right) > 0
        for i in range(len(left)):
            x, y = left[i], right[i]
            for key in ('raw_path', 'source_recorded_raw_path'):
                x[key] = y[key] = 'normalized-output-root'
            assert pack_record(x) == pack_record(y)
    finally:
        left.close(); right.close()


def test_compressed_storage_roundtrip_and_corruption(tmp_path):
    import zlib
    from apexgen.shared.storage.store import pack_record, unpack_record
    record = adapt(tmp_path, native_arrays(source_arrays()))
    packed = pack_record(record, compression='zlib')
    assert pack_record(unpack_record(packed)) == pack_record(record)
    assert len(packed) < len(pack_record(record))
    with pytest.raises(zlib.error):
        unpack_record(packed[:-3])
