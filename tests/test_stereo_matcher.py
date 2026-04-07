"""Tests for stereo matcher foundation backend integration behavior."""

import pathlib
import sys
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import cv2  # type: ignore
except Exception:
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.COLOR_BGR2GRAY = 0
    cv2_stub.COLOR_GRAY2RGB = 0
    cv2_stub.STEREO_SGBM_MODE_SGBM_3WAY = 0
    cv2_stub.CV_32F = 5
    cv2_stub.StereoSGBM_create = lambda *args, **kwargs: None
    cv2_stub.ximgproc = types.SimpleNamespace(
        createRightMatcher=lambda *args, **kwargs: None,
        createDisparityWLSFilter=lambda *args, **kwargs: types.SimpleNamespace(
            setLambda=lambda *_a, **_k: None,
            setSigmaColor=lambda *_a, **_k: None,
            filter=lambda **_kwargs: None,
        ),
    )
    cv2_stub.cvtColor = lambda img, _code: img
    cv2_stub.Sobel = lambda arr, *_args, **_kwargs: arr
    sys.modules['cv2'] = cv2_stub

from depth.stereo_matcher import StereoMatcher, StereoMethod


class DummyModule:
    """Minimal module-like model used to test loader wiring."""

    def __init__(self):
        self.eval_called = False

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.eval_called = True
        return self


class TestFoundationLoader(unittest.TestCase):
    @staticmethod
    def _has_torch():
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def test_invalid_loader_spec_falls_back_to_sgbm(self):
        with mock.patch.object(StereoMatcher, '_init_sgbm', autospec=True) as init_sgbm:
            matcher = StereoMatcher({
                'method': 'foundation',
                'foundation_model_path': '/tmp/model.ckpt',
                'foundation_loader': 'badformat',
            })

        self.assertEqual(matcher.method, StereoMethod.SGBM)
        init_sgbm.assert_called()

    def test_loader_spec_resolves_and_loads_model(self):
        module_name = 'tests.fake_foundation_loader'
        fake_module = types.ModuleType(module_name)

        def load_model(path):
            self.assertEqual(path, '/tmp/model.ckpt')
            return DummyModule()

        fake_module.load_model = load_model
        sys.modules[module_name] = fake_module

        try:
            with mock.patch.object(StereoMatcher, '_init_sgbm', autospec=True) as init_sgbm:
                matcher = StereoMatcher({
                    'method': 'foundation',
                    'foundation_model_path': '/tmp/model.ckpt',
                    'foundation_loader': f'{module_name}:load_model',
                    'foundation_compile': False,
                    'foundation_use_half': False,
                    'foundation_channels_last': False,
                    'foundation_use_amp': False,
                })

            if self._has_torch():
                self.assertEqual(matcher.method, StereoMethod.FOUNDATION)
                self.assertTrue(hasattr(matcher, 'foundation_model'))
                self.assertTrue(matcher.foundation_model.eval_called)
                init_sgbm.assert_not_called()
            else:
                self.assertEqual(matcher.method, StereoMethod.SGBM)
                init_sgbm.assert_called()
        finally:
            sys.modules.pop(module_name, None)

    def test_foundation_fast_mode_sets_fast_defaults(self):
        module_name = 'tests.fake_foundation_loader_fast'
        fake_module = types.ModuleType(module_name)
        fake_module.load_model = lambda _path: DummyModule()
        sys.modules[module_name] = fake_module

        try:
            with mock.patch.object(StereoMatcher, '_init_sgbm', autospec=True):
                matcher = StereoMatcher({
                    'method': 'foundation_fast',
                    'foundation_model_path': '/tmp/model.ckpt',
                    'foundation_loader': f'{module_name}:load_model',
                    'foundation_compile': False,
                })

            if self._has_torch():
                self.assertEqual(matcher.method, StereoMethod.FOUNDATION_FAST)
                self.assertTrue(matcher.foundation_use_half)
                self.assertTrue(matcher.foundation_channels_last)
            else:
                self.assertEqual(matcher.method, StereoMethod.SGBM)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == '__main__':
    unittest.main()
