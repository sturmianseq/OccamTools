import os
import pytest
import numpy as np
import warnings
import shutil
from occamtools.read_fort1 import Fort1
from occamtools.read_fort7 import Fort7
from occamtools.read_xyz import Xyz, _are_floats
from occamtools.occam_data import (OccamData, _check_internal_consistency_all,
                                   _check_internal_consistency)


file_name_fort_1 = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                'fort.1')
file_name_fort_7 = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                'fort.7')
file_name_fort_xyz = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                  'fort.8')
class_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                         'class_data')
ignore = ['file_name', 'n_time_steps_', 'file_contents',
          'comment_format_known', 'num_lines']


def _load_default_forts(silent=True):
    fort1 = Fort1(file_name_fort_1)
    fort1.read_file(silent=silent)
    fort7 = Fort7(file_name_fort_7)
    fort7.read_file(silent=silent)
    xyz = Xyz(file_name_fort_xyz)
    xyz.read_file(silent=silent)
    return fort1, fort7, xyz


def _create_default_occam_data_object(load_from_npy=False, silent=True):
    fort1, fort7, xyz = _load_default_forts(silent=silent)
    occam_data = OccamData(fort1, fort7, xyz, load_from_npy=load_from_npy,
                           silent=silent)
    return occam_data, fort1, fort7, xyz


def _assert_all_attributes_present(occam_data, fort1, fort7, xyz):
    for f in (fort1, fort7, xyz):
        for key in f.__dict__:
            assert key in occam_data.__dict__ if key not in ignore else True


def _check_equal(a, b):
    if _are_floats(a, b):
        return float(a) == pytest.approx(float(b), abs=1e-14)
    elif isinstance(a, str) and isinstance(b, str):
        return a == b
    elif hasattr(a, 'shape') and hasattr(b, 'shape'):
        return np.allclose(a, b)
    else:
        return a == b


def _assert_all_attributes_equal(occam_data, fort1, fort7, xyz):
    for f in (fort1, fort7, xyz):
        for key in f.__dict__:
            if key not in ignore:
                assert _check_equal(f.__dict__[key], occam_data.__dict__[key])


def _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz):
    _assert_all_attributes_present(occam_data, fort1, fort7, xyz)
    _assert_all_attributes_equal(occam_data, fort1, fort7, xyz)


def test_occam_data_check_internal_consistency_1_7():
    fort1, fort7, xyz = _load_default_forts()

    c17 = _check_internal_consistency(fort1, fort7)
    assert c17
    c1x = _check_internal_consistency(fort1, xyz)
    assert c1x
    c7x = _check_internal_consistency(fort7, xyz)
    assert c7x

    all_ = _check_internal_consistency_all(fort1, fort7, xyz)
    assert all_
    assert (c17 and c1x and c7x) is all_


def test_occam_data_constructor_files():
    fort1, fort7, xyz = _load_default_forts()
    occam_data = OccamData(fort1.file_name, fort7.file_name, xyz.file_name,
                           silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(fort1.file_name, None, None, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(None, fort7.file_name, None, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(None, None, xyz.file_name, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)


def test_occam_data_attributes():
    occam_data, fort1, fort7, xyz = _create_default_occam_data_object()
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    shutil.rmtree(class_dir)


def test_occam_data_wrong_input():
    caught = False
    try:
        _ = OccamData(None, None, None, silent=True)
    except ValueError:
        caught = True
    assert caught


def test_occam_data_not_equal():
    for i in range(2):
        fort1, fort7, xyz = _load_default_forts()
        if i == 0:
            fort1.n_particles += 1
        else:
            fort1.title = 'changed'

        warnings.filterwarnings('ignore')
        fort1.n_particles += 1
        occam_data = OccamData(fort1, fort7, xyz, silent=True)
        assert occam_data.consistent is False

        warnings.filterwarnings('error')
        caught = False
        try:
            _ = OccamData(fort1, fort7, xyz, silent=True)
        except Warning:
            caught = True
        assert caught
    warnings.filterwarnings('always')


def test_occam_data_single_input():
    fort1, fort7, xyz = _load_default_forts()
    occam_data = OccamData(fort1.file_name, load_from_npy=False, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(fort7.file_name, load_from_npy=False, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(xyz.file_name, load_from_npy=False, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)
    occam_data = OccamData(os.path.dirname(fort1.file_name),
                           load_from_npy=False, silent=True)
    _assert_all_attributes_present_and_equal(occam_data, fort1, fort7, xyz)

    inputs = ['this_is_not_a_file', None, 1, 8.29898, fort1, fort7, xyz]
    for inp in inputs:
        caught = False
        try:
            _ = OccamData(inp, load_from_npy=False, silent=True)
        except TypeError:
            caught = True
        except FileNotFoundError:
            caught = True
        assert caught


def test_occam_data_save_load():
    shutil.rmtree(class_dir, ignore_errors=True)
    occam_data, fort1, fort7, xyz = _create_default_occam_data_object()
    all_attributes = [key for key in occam_data.__dict__]
    attributes = []
    for key in all_attributes:
        if isinstance(occam_data.__dict__[key], np.ndarray):
            attributes.append(key)
    occam_data.save()

    for array in attributes:
        expected_file_name = array + '.npy'
        assert os.path.exists(os.path.join(class_dir, expected_file_name))
    for key in all_attributes:
        assert key in occam_data.__dict__

    occam_data_npy_loaded = OccamData(os.path.join(class_dir, os.pardir),
                                      silent=True)
    for key in all_attributes:
        assert key in occam_data_npy_loaded.__dict__

    assert not occam_data_npy_loaded.save(overwrite=False)
    occam_data_npy_loaded_file = OccamData(os.path.join(class_dir, os.pardir,
                                                        'fort.1'),
                                           silent=True)
    _assert_all_attributes_present_and_equal(occam_data_npy_loaded_file,
                                             fort1, fort7, xyz)
    caught = False
    try:
        _ = OccamData('this_is_not_a_file', silent=True)
    except FileNotFoundError:
        caught = True
    assert caught

    shutil.rmtree(class_dir)


def test_occam_data_progress_bars():
    fort1, fort7, fort8 = _load_default_forts()
    occam_data_silent = OccamData(fort1, fort7, fort8)
    fort1, fort7, fort8 = _load_default_forts(silent=False)
    occam_data_verbose = OccamData(fort1, fort7, fort8)
    for key in occam_data_silent.__dict__:
        assert _check_equal(occam_data_silent.__dict__[key],
                            occam_data_verbose.__dict__[key])
    occam_data_verbose.save()
    occam_data_verbose_npy = OccamData(file_name_fort_1, load_from_npy=False,
                                       silent=True)
    occam_data_verbose_npy = OccamData(file_name_fort_1, load_from_npy=True,
                                       silent=False)
    for key in occam_data_silent.__dict__:
        assert _check_equal(occam_data_silent.__dict__[key],
                            occam_data_verbose_npy.__dict__[key])
    shutil.rmtree(class_dir)


def test_occam_data_not_save_to_npy():
    assert not os.path.exists(class_dir)
    _ = OccamData(file_name_fort_1, save_to_npy=False, silent=True)
    assert not os.path.exists(class_dir)
