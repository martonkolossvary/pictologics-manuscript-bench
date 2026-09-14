from dataclasses import replace
import json
from pathlib import Path

import pytest

from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError, RunSpecMismatch, atomic_write_json
from bench.benchmark_memory import MemoryPreflightPolicy, LEGACY_MEMORY_POLICY_ID
from bench.benchmark_models import task_plan_fingerprint
from bench.benchmark_recovery import corrected_spec, plan_repair, apply_repair, sync_recovery_exports
from tests.test_benchmark_ledger import make_task, make_run_spec


def legacy_policy():
    policy=MemoryPreflightPolicy(user_cap_bytes=12*2**30).to_dict()
    policy.update(policy_id=LEGACY_MEMORY_POLICY_ID,
        empirical_scope={'primary':['adapter','workload','guardrail_group'],
                         'cross_stratum_upper':['adapter','workload']})
    policy.pop('empirical_anchor');policy.pop('empirical_baseline')
    return policy


@pytest.fixture
def recovery_fixture(tmp_path,monkeypatch):
    measured=make_task()
    rejected=replace(measured,ordinal=2,case_id='case-2',image_voxels=2000)
    skipped=replace(measured,ordinal=3,case_id='case-3',image_voxels=4000)
    actual=replace(measured,ordinal=4,case_id='other-case',subject_id='other')
    tasks=[measured,rejected,skipped,actual]
    original=replace(make_run_spec(measured),memory_preflight=tuple(legacy_policy().items()),
        task_plan_sha256=task_plan_fingerprint(tasks),selected_case_ids=tuple(t.case_id for t in tasks))
    ledger=BenchmarkLedger(tmp_path/'benchmark.sqlite3')
    ledger.initialize(original,tasks,resume=False)
    record={'task_id':measured.task_id,'task_status':'measured','duration_sec':1.0}
    payload={'benchmark':{'task_id':measured.task_id,'status':'measured',
                          'run_fingerprint':original.run_fingerprint,'record':record}}
    digest=atomic_write_json(tmp_path/'payload.json',payload)
    ledger.mark_running(measured.task_id)
    ledger.mark_terminal(measured.task_id,'measured',record,duration_sec=1.0,payload_path='payload.json',payload_sha256=digest)
    for task,status,evidence in [(rejected,'skipped_memory_preflight',rejected),
                                 (skipped,'skipped_memory_cutoff',rejected),
                                 (actual,'memory_limit_exceeded',actual)]:
        rec={'task_id':task.task_id,'task_status':status,'memory_preflight_policy_id':LEGACY_MEMORY_POLICY_ID,
             'memory_cutoff_evidence_task_id':evidence.task_id}
        if task==actual:ledger.mark_running(task.task_id)
        ledger.mark_terminal(task.task_id,status,rec,error='memory evidence')
    for task in (rejected,actual):
        ledger.activate_memory_cutoff(scope_key=task.case_id,adapter=task.adapter,workload_key=task.workload,
            guardrail_group=task.guardrail_group,complexity_metric='image_voxels',cutoff_complexity=task.image_voxels,
            reason='memory evidence',evidence_task_id=task.task_id)
    atomic_write_json(tmp_path/'run_spec.json',original.to_dict())
    monkeypatch.setattr('bench.benchmark_recovery.source_evidence',lambda *args:{'new_sources_sha256':'new-source'})
    monkeypatch.setattr('bench.run._benchmark_sources_sha256',lambda *args,**kwargs:'new-source')
    yield ledger,original,tasks
    ledger.close()


def test_repair_preserves_payloads_and_attempts_and_requeues_only_estimates(recovery_fixture,tmp_path):
    ledger,old,tasks=recovery_fixture
    before=(tmp_path/'payload.json').read_bytes()
    measured_before=dict(ledger.task_row(tasks[0].task_id))
    attempts=[tuple(r) for r in ledger.connection.execute('SELECT * FROM task_attempts ORDER BY task_id,attempt')]
    plan=plan_repair(ledger,Path.cwd())
    assert len(plan['requeue'])==2
    result=apply_repair(ledger,plan,tmp_path/'backup')
    assert result['requeued']==2
    assert (tmp_path/'payload.json').read_bytes()==before
    assert dict(ledger.task_row(tasks[0].task_id))==measured_before
    assert attempts==[tuple(r) for r in ledger.connection.execute('SELECT * FROM task_attempts ORDER BY task_id,attempt')]
    assert [ledger.status(t.task_id) for t in tasks]==['measured','pending','pending','memory_limit_exceeded']
    assert ledger.connection.execute('SELECT COUNT(*) FROM memory_cutoffs').fetchone()[0]==1
    records,payloads=ledger.verified_records_and_payloads()
    assert payloads[0]['benchmark']['run_fingerprint']==old.run_fingerprint
    assert ledger.measurement_origins()[tasks[0].task_id]==old.run_fingerprint
    assert len(ledger.protocol_amendments())==1
    revised=corrected_spec(old.to_dict(),'new-source')
    ledger.initialize(revised,tasks,resume=True)
    with pytest.raises(RunSpecMismatch):
        ledger.initialize(replace(revised,timeout_seconds=123),tasks,resume=True)
    assert plan_repair(ledger,Path.cwd())['already_repaired']


def test_corrupt_measurement_rolls_back_requeue_and_amendment(recovery_fixture,tmp_path):
    ledger,old,tasks=recovery_fixture
    plan=plan_repair(ledger,Path.cwd())
    (tmp_path/'payload.json').write_text('tampered',encoding='utf-8')
    with pytest.raises(RunIntegrityError,match='checksum'):
        apply_repair(ledger,plan,tmp_path/'backup')
    assert ledger.metadata_value('run_fingerprint')==old.run_fingerprint
    assert ledger.status(tasks[1].task_id)=='skipped_memory_preflight'
    assert ledger.protocol_amendments()==[]
    assert ledger.connection.execute('SELECT COUNT(*) FROM measurement_origins').fetchone()[0]==0


def test_recovery_rejects_running_tasks(recovery_fixture):
    ledger,old,tasks=recovery_fixture
    ledger.connection.execute("UPDATE tasks SET status='running',record_json=NULL WHERE task_id=?",(tasks[1].task_id,))
    ledger.connection.commit()
    with pytest.raises(RunIntegrityError,match='running task'):
        plan_repair(ledger,Path.cwd())


def test_historical_payload_cannot_be_accepted_without_its_origin(recovery_fixture,tmp_path):
    ledger,_,_=recovery_fixture
    apply_repair(ledger,plan_repair(ledger,Path.cwd()),tmp_path/'backup')
    ledger.connection.execute('DELETE FROM measurement_origins');ledger.connection.commit()
    with pytest.raises(RunIntegrityError,match='identity'):
        ledger.verified_records_and_payloads()


def test_amendment_tampering_is_rejected(recovery_fixture,tmp_path):
    ledger,_,_=recovery_fixture
    apply_repair(ledger,plan_repair(ledger,Path.cwd()),tmp_path/'backup')
    ledger.connection.execute("UPDATE protocol_amendments SET old_fingerprint='forged'");ledger.connection.commit()
    with pytest.raises(RunIntegrityError,match='fingerprint chain'):
        ledger.verified_records_and_payloads()


def test_exports_can_finish_after_transaction_committed(recovery_fixture,tmp_path):
    ledger,old,_=recovery_fixture
    plan=plan_repair(ledger,Path.cwd())
    apply_repair(ledger,plan,tmp_path/'backup')
    assert json.loads((tmp_path/'run_spec.json').read_text())==old.to_dict()
    sync_recovery_exports(ledger,Path.cwd())
    assert json.loads((tmp_path/'run_spec.json').read_text())==plan['new_spec']
    sync_recovery_exports(ledger,Path.cwd())
    assert json.loads((tmp_path/'run_meta.json').read_text())['derived_exports_stale']


def test_unknown_legacy_memory_settings_fail_closed(recovery_fixture):
    _,old,_=recovery_fixture
    changed=old.to_dict();changed['memory_preflight']['invented_setting']=True
    with pytest.raises(RunIntegrityError,match='unsupported changes'):
        corrected_spec(changed,'new-source')
