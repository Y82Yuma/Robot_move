import sys
import traceback
import joblib
import numpy as np
from pathlib import Path
import sys
from pathlib import Path as _Path

def safe_get(obj, attr):
    try:
        return getattr(obj, attr)
    except Exception:
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tmp_check_model.py <model_path>')
        sys.exit(2)
    p = Path(sys.argv[1])
    print('Model path:', p)
    if not p.exists():
        print('Model file not found')
        sys.exit(1)
    # Ensure local project src and affetto-nn-ctrl/src are on sys.path so
    # joblib unpickling can import affetto_nn_ctrl and related modules.
    try:
        proj_root = _Path(__file__).resolve().parent
        candidates = [proj_root / 'src', proj_root.parent / 'affetto-nn-ctrl' / 'src']
        for c in candidates:
            cs = str(c)
            if c.exists() and cs not in sys.path:
                sys.path.insert(0, cs)
    except Exception:
        pass

    try:
        m = joblib.load(str(p))
    except Exception as e:
        print('joblib.load failed:')
        traceback.print_exc()
        sys.exit(1)
    print('Loaded object type:', type(m))
    try:
        print('repr(m):', repr(m)[:1000])
    except Exception:
        pass
    print('Has predict?:', hasattr(m, 'predict'))
    # Inspect possible wrapped model
    model_attr = safe_get(m, 'model')
    print('Has .model attribute?:', model_attr is not None)
    if model_attr is not None:
        print('  model type:', type(model_attr))
        print('  model repr:', repr(model_attr)[:1000])
        print('  model has predict?:', hasattr(model_attr, 'predict'))
        print('  model n_features_in_?:', safe_get(model_attr, 'n_features_in_'))
    print('Top-level n_features_in_?:', safe_get(m, 'n_features_in_'))
    # adapter
    adapter = safe_get(m, 'adapter')
    print('Has adapter?:', adapter is not None)
    if adapter is not None:
        print(' adapter type:', type(adapter))
        params = safe_get(adapter, 'params')
        print(' adapter.params:', params)
        if params is not None:
            try:
                for k in ('active_joints','dof','include_tension','angle_unit'):
                    print('  params.'+k+':', safe_get(params, k))
            except Exception:
                pass
    # Decide n_features to test
    nfeat = 1
    if isinstance(model_attr, object) and safe_get(model_attr, 'n_features_in_') is not None:
        nfeat = int(getattr(model_attr, 'n_features_in_'))
    elif safe_get(m, 'n_features_in_') is not None:
        nfeat = int(getattr(m, 'n_features_in_'))
    else:
        # Try pipeline scaler
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(model_attr, Pipeline):
                scaler = model_attr.steps[0][1]
                if safe_get(scaler, 'n_features_in_') is not None:
                    nfeat = int(getattr(scaler, 'n_features_in_'))
        except Exception:
            pass
    print('Will test predict with n_features =', nfeat)
    X = np.zeros((1, nfeat), dtype=float)
    try:
        y = m.predict(X)
        print('Predict succeeded. type(y)=', type(y))
        import numpy as _np
        y_arr = _np.asarray(y)
        print('y_arr.shape =', getattr(y_arr, 'shape', None))
        print('y_arr dtype =', getattr(y_arr, 'dtype', None))
        print('y sample:', y_arr.ravel()[:10])
    except Exception as e:
        print('Predict failed:')
        traceback.print_exc()
        sys.exit(0)
    sys.exit(0)
