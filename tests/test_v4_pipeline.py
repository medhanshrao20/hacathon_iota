"""Small synthetic CLI run: all policies, freeze order, stress separation and inference."""
import json
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from sentinel.pipeline import ROOT
from sentinel.models import Forecaster
from sentinel.policies import fit_policy, replay_policy


class ControlledPipeline(unittest.TestCase):
    def test_synthetic_pipeline_is_portable_and_test_changes_cannot_change_selection(self):
        config=json.loads((ROOT/"config.json").read_text())
        control=json.loads((ROOT/"controls/v2_config.json").read_text())
        control.update(windows=[8],alphas=[10.],horizon_sets=[[1],[1,4]],threshold_quantiles=[.99],
                       adaptation=[False],persistence=[[2,3]],threshold_methods=["empirical"],
                       incident_profiles=["guarded"],max_seasonal_period=64,
                       min_calibration_rows=16,reference_size=32)
        config["v4"].update(bootstrap_repetitions=100,runtime_windows=5,secondary_window=8,
                            injection_seeds=[17,83],injection_lengths=[4,8])
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            data=folder/"data"
            data.mkdir()
            control_file=folder/"control.json"
            control_file.write_text(json.dumps(control))
            stamps=pd.date_range("2026-01-01",periods=800,freq="min",tz="UTC")
            values=np.sin(np.arange(800)/12.)+.05*np.random.default_rng(42).normal(size=800)
            values[480:485]+=5
            values[720:725]+=5
            labels={}
            for name in ("one","two"):
                (data/name).mkdir()
                pd.DataFrame(dict(timestamp=stamps,value=values)).to_csv(data/name/"stream.csv",index=False)
                labels[name+"/stream.csv"]=[[str(stamps[480]),str(stamps[484])],
                                            [str(stamps[720]),str(stamps[724])]]
            label_file=folder/"labels.json"
            label_file.write_text(json.dumps(labels))
            config.update(control_config=str(control_file),data_dir=str(data),labels_file=str(label_file),
                          output_dir=str(folder/"results"))
            config_file=folder/"config.json"
            config_file.write_text(json.dumps(config))
            def run(*args):
                result=subprocess.run([sys.executable,str(ROOT/"run.py"),"--config",str(config_file),*args],
                                      capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run()
            first=next((folder/"results").glob("experiment_v4_*"))
            frozen=first/"selection_frozen_before_test.json"
            selection=json.loads(frozen.read_text())
            records=pd.read_csv(first/"comparison.csv")
            self.assertEqual(len(records),18)
            self.assertEqual(int(records.selected_on_validation.sum()),2)
            self.assertTrue(all(p.stat().st_mtime>=frozen.stat().st_mtime for p in first.glob("series/*/predictions.csv")))
            self.assertGreaterEqual((first/"synthetic_stress.csv").stat().st_mtime,frozen.stat().st_mtime)
            stress=pd.read_csv(first/"synthetic_stress.csv")
            self.assertFalse(stress.used_for_selection.any())
            self.assertEqual(set(stress.kind),{"spike","dip","flatline","level","variance","drift"})
            self.assertTrue(pd.read_csv(first/"runtime_per_window.csv").per_window_p95_ms.gt(0).all())
            artifact=next(first.glob("series/*/model.joblib"))
            self.assertEqual(joblib.load(artifact)["format_version"],4)
            new=folder/"later.csv"
            pd.DataFrame(dict(timestamp=pd.date_range(stamps[-1]+pd.Timedelta(minutes=1),periods=10,freq="min"),
                              value=np.zeros(10))).to_csv(new,index=False)
            run("--predict",str(new),"--artifact",str(artifact))
            self.assertEqual(len(pd.read_csv(next((folder/"results").glob("predict_*/predictions.csv")))),10)
            # Exercise saved confirmation state as well as the conservative selected control.
            bundle=joblib.load(artifact)
            secondary=Forecaster("pattern",8,(1,)).fit(values,np.ones(800,bool),400,config=config)
            predictions=bundle["model"].predict_aligned(values)
            distances=secondary.auxiliary_scores(values)
            errors=values[:,None]-predictions
            detector,_=fit_policy(errors[400:560],bundle["parameters"],bundle["raw_floor"],config,
                dict(name="F_confirmation_bypass",revised=True,spike=True,cusum=False,confirm=True),distances[400:560])
            replay_policy(detector,values,predictions,np.ones(800,bool),560,800,secondary=distances)
            bundle.update(detector=detector,secondary_model=secondary,history_values=values[-16:],
                          history_observed=np.ones(16,bool),history_timestamps=stamps[-16:].astype(str).tolist())
            confirmed_artifact=folder/"confirmed.joblib"
            joblib.dump(bundle,confirmed_artifact)
            run("--predict",str(new),"--artifact",str(confirmed_artifact))
            confirmed_path=sorted((folder/"results").glob("predict_*/predictions.csv"))[-1]
            confirmed=pd.read_csv(confirmed_path)
            self.assertTrue(confirmed.confirmation_threshold.eq(config["v4"]["confirmation_percentile"]).all())
            self.assertTrue(confirmed.evidence.str.contains("secondary percentile/cutoff").all())
            extended=np.r_[values,np.zeros(10)]
            expected=replay_policy(copy.deepcopy(detector),extended,bundle["model"].predict_aligned(extended),
                np.ones(810,bool),800,810,secondary=secondary.auxiliary_scores(extended))
            np.testing.assert_array_equal(confirmed.label.eq("ANOMALOUS"),expected["flag"])
            # Keep every development value/label unchanged, but replace the complete test suffix.
            changed=values.copy()
            changed[680:]=1e8
            for name in ("one","two"):
                pd.DataFrame(dict(timestamp=stamps,value=changed)).to_csv(data/name/"stream.csv",index=False)
                labels[name+"/stream.csv"]=labels[name+"/stream.csv"][:1]+[[str(stamps[690]),str(stamps[790])]]
            label_file.write_text(json.dumps(labels))
            run()
            second=sorted((folder/"results").glob("experiment_v4_*"))[-1]
            self.assertEqual(selection,json.loads((second/"selection_frozen_before_test.json").read_text()))
