import os
from copy import deepcopy
import numpy as np


class _Properties:
    ATOM_TYPE = 0
    BOND_TYPE = 1
    BOND_ANGLE = 2
    TORSION = 3
    NON_BONDED = 4
    SCF = 5
    COMPRESSIBILITY = 6
    CHI = 7

    @staticmethod
    def _type_from_index(index):
        for v in _Properties.__dict__:
            if not v.startswith('__'):
                if _Properties.__dict__[v] == index:
                    return v
        raise IndexError(f'Property index {index} not valid. Expected:\n'
                         '0 <= index <= 7')


class Fort3Replacement:
    def __init__(self, property=None, replace=None, new=None, content=None):
        self._content = content

        if property is not None:
            self._parse_property_name(property)
        else:
            self.property = property

        if isinstance(replace, bool) and isinstance(new, bool):
            if replace is new:
                raise ValueError('Replace and new cannot both be ' + str(new))
            else:
                self._new = new
                self._replace = replace
        elif isinstance(replace, bool) or isinstance(new, bool):
            if isinstance(replace, bool):
                self._replace = replace
                self._new = not replace
            else:
                self._new = new
                self._replace = not new
        elif (new is None) and (replace is None):
            self._new = new
            self._replace = replace
        else:
            raise TypeError('Expected None or bool for new and replace, not '
                            + str(type(new)) + ' and ' + str(type(replace)))

    @property
    def new(self):
        return self._new

    @new.setter
    def new(self, new):
        if not isinstance(new, bool):
            raise TypeError('Expected bool for new, got ' + str(type(new)))
        self._new = new
        self._replace = not new

    @property
    def replace(self):
        return self._replace

    @replace.setter
    def replace(self, replace):
        if not isinstance(replace, bool):
            raise TypeError('Expected bool for replace, got '
                            + str(type(replace)))
        self._replace = replace
        self._new = not replace

    def _parse_property_name(self, property):
        p = property.lower()
        if 'atom' in p:
            self.property = _Properties.ATOM_TYPE
        elif 'bond' in p and 'type' in p:
            self.property = _Properties.BOND_TYPE
        elif 'angle' in p:
            self.property = _Properties.BOND_ANGLE
        elif 'torsion' in p:
            self.property = _Properties.TORSION
        elif 'non' in p and 'bond' in p:
            self.property = _Properties.NON_BONDED
        elif 'scf' in p or 'hpf' in p:
            self.property = _Properties.SCF
        elif 'comp' in p:
            self.property = _Properties.COMPRESSIBILITY
        elif 'chi' in p:
            self.property = _Properties.CHI
        else:
            error_string = ('Property string ' + property + ' was not'
                            ' recognized')
            raise ValueError(error_string)

    def __repr__(self):
        prop_type = _Properties._type_from_index(self.property)
        return (f'Fort3Replacement(property={prop_type}, '
                f'replace={self._replace}, new={self._new}, '
                f'content={self._content})')


def _count_property_instances(*args):
    counts = {key: 0 for key in range(5)}
    for p in args:
        if (p.new is True) and (p.property <= 4):
            counts[p.property] += 1
    return counts


def _is_int(s):
    try:
        s = float(s)
        if s.is_integer():
            return True
        else:
            return False
    except ValueError:
        return False


def _count_existing_instances(fort_file):
    keys = ['atom types', 'bond types', 'bond angles', 'torsions', 'non-bond']
    counts = {key: 0 for key in range(5)}
    with open(fort_file, 'r') as in_file:
        for line in in_file:
            sline = line.split()
            if len(sline) > 1:
                if _is_int(sline[0]):
                    for i, key in enumerate(keys):
                        if key in line:
                            counts[i] = int(sline[0])
                            break
    return counts


def _parse_fort_3_file(fort_file):
    atom_name = {}
    atoms = []
    bonds = []
    angles = []
    torsions = []
    non_bonds = []

    with open(fort_file, 'r') as in_file:
        for _ in range(4):
            line = in_file.readline()
        while '*****' not in line:  # atom types
            index, name, mass, charge = line.split()
            index, mass, charge = int(index), float(mass), float(charge)
            atom_name[index] = name
            atoms.append(Fort3Replacement(property='atom', new=True,
                                          content=[name, mass, charge]))
            line = in_file.readline()

        for _ in range(3):
            line = in_file.readline()
        while '******' not in line:  # bond types
            index_1, index_2, sigma, eps = line.split()
            index_1, index_2 = int(index_1), int(index_2)
            sigma, eps = float(sigma), float(eps)
            content = [atom_name[index_1], atom_name[index_2], sigma, eps]
            bonds.append(Fort3Replacement(property='bond type', new=True,
                                          content=content))
            line = in_file.readline()

        for _ in range(3):
            line = in_file.readline()
        while '******' not in line:  # bond angles
            ind1, ind2, ind3, theta, eps = line.split()
            ind1, ind2, ind3 = (int(i) for i in (ind1, ind2, ind3))
            theta, eps = float(theta), float(eps)
            content = [atom_name[ind1], atom_name[ind2], atom_name[ind3],
                       theta, eps]
            angles.append(Fort3Replacement(property='bond angle', new=True,
                                           content=content))
            line = in_file.readline()

        for _ in range(3):
            line = in_file.readline()
        while '******' not in line:  # torsions
            ind1, ind2, ind3, ind4, phi, eps = line.split()
            ind1, ind2, ind3, ind4 = (int(i) for i in (ind1, ind2, ind3, ind4))
            phi, eps = float(phi), float(eps)
            content = [atom_name[ind1], atom_name[ind2], atom_name[ind3],
                       atom_name[ind4], phi, eps]
            torsions.append(Fort3Replacement(property='bond angle', new=True,
                                             content=content))
            line = in_file.readline()

        for _ in range(3):
            line = in_file.readline()
        while '******' not in line:  # non-bonded interactions
            index_1, index_2, sigma, eps = line.split()
            index_1, index_2 = int(index_1), int(index_2)
            sigma, eps = float(sigma), float(eps)
            content = [atom_name[index_1], atom_name[index_2], sigma, eps]
            non_bonds.append(Fort3Replacement(property='non bonded', new=True,
                                              content=content))
            line = in_file.readline()

        for _ in range(2):
            line = in_file.readline()
        scf = [int(i) for i in line.split()]
        for _ in range(2):
            line = in_file.readline()
        kappa = float(line)
        line = in_file.readline()
        chi = np.empty(shape=(len(atom_name), len(atom_name)))
        for i in range(chi.shape[1]):
            sline = in_file.readline().split()
            chi[i, :] = np.asarray([float(s) for s in sline])

        return (atom_name, atoms, bonds, angles, torsions, non_bonds, scf,
                kappa, chi)


def _sort_new_replace_args_atom(atom_names_, atoms_, *args):
    atom_names = deepcopy(atom_names_)
    atoms = deepcopy(atoms_)
    for arg in args:
        if len(arg._content) != 3:
            error_str = (f'Invalid content for replacement object {repr(arg)},'
                         f' the content kwarg must have length 3, not '
                         f'{len(arg._content)}')
            raise ValueError()
        if arg.property == _Properties.ATOM_TYPE:
            name = arg._content[0]
            found = False
            index = None
            for i, atom in enumerate(atoms):
                if name == atom._content[0]:
                    found = True
                    index = i
                    break
            if found and (arg.replace is True):
                atoms[index] = arg
            elif found and (arg.new is True):
                error_str = (f'Cannot add atom type {repr(arg)}, a type of '
                             f'this name already exists')
                raise ValueError(error_str)
            elif (not found) and (arg.new is True):
                atoms.append(arg)
                atom_names[max(atom_names)+1] = arg._content[0]
            elif (not found) and (arg.replace is True):
                error_str = (f'No existing atom with name {name} was '
                             f'found to replace with {repr(args)}')
                raise ValueError(error_str)
    return atom_names, atoms


def _sort_new_replace_args_bonds(atom_names_, bonds_, *args):
    atom_names = deepcopy(atom_names_)
    bonds = deepcopy(bonds_)
    for arg in args:
        if len(arg._content) != 4:
            error_str = (f'Invalid content for replacement object {repr(arg)},'
                         f' the content kwarg must have length 4, not '
                         f'{len(arg._content)}')
            raise ValueError()
        if arg.property == _Properties.BOND_TYPE:
            name_1, name_2 = arg._content[:2]
            if name_1 not in atom_names.values():
                error_str = (f'Cannot establish bond type {repr(arg)}, '
                             f'between atoms {name_1} and {name_2}. {name_1} '
                             f'not found in atom dict: {atom_names}.')
                raise ValueError(error_str)
            elif name_2 not in atom_names.values():
                error_str = (f'Cannot establish bond type {repr(arg)}, '
                             f'between atoms {name_1} and {name_2}. {name_2} '
                             f'not found in atom dict: {atom_names}.')
                raise ValueError(error_str)

            found = False
            index = None
            for i, bond in enumerate(bonds):
                n1, n2 = bond._content[:2]
                if (((name_1 == n1) and (name_2 == n2)) or
                        ((name_1 == n2) and (name_2 == n1))):
                    found = True
                    index = i
                    break
            if found and (arg.replace is True):
                bonds[index] = arg
            elif found and (arg.new is True):
                error_str = (f'Cannot add bond type {repr(arg)}, a bond '
                             f' between {name_1} and {name_2} already exists')
                raise ValueError(error_str)
            elif (not found) and (arg.new is True):
                bonds.append(arg)
            elif (not found) and (arg.replace is True):
                error_str = (f'No existing bond between {name_1} and {name_2} '
                             f'was found to replace with {repr(arg)}')
                raise ValueError(error_str)
    return bonds


def _sort_new_replace_args_angles(atom_names_, angles_, *args):
    atom_names = deepcopy(atom_names_)
    angles = deepcopy(angles_)
    for arg in args:
        if len(arg._content) != 5:
            error_str = (f'Invalid content for replacement object {repr(arg)},'
                         f' the content kwarg must have length 4, not '
                         f'{len(arg._content)}')
            raise ValueError()
        if arg.property == _Properties.BOND_ANGLE:
            name_1, name_2, name_3 = arg._content[:3]
            if name_1 not in atom_names.values():
                error_str = (f'Cannot establish bond angle {repr(arg)}, '
                             f'between atoms {name_1}, {name_2}, and {name_3}.'
                             f' {name_1} not found in atom dict: {atom_names}')
                raise ValueError(error_str)
            elif name_2 not in atom_names.values():
                error_str = (f'Cannot establish bond angle {repr(arg)}, '
                             f'between atoms {name_1}, {name_2}, and {name_3}.'
                             f' {name_2} not found in atom dict: {atom_names}')
                raise ValueError(error_str)
            elif name_3 not in atom_names.values():
                error_str = (f'Cannot establish bond angle {repr(arg)}, '
                             f'between atoms {name_1}, {name_2}, and {name_3}.'
                             f' {name_3} not found in atom dict: {atom_names}')
                raise ValueError(error_str)
            found = False
            index = None
            for i, angle in enumerate(angles):
                n1, n2, n3 = angle._content[:3]
                if n2 == name_2:
                    if (name_1 == n1) and (name_3 == n3):
                        found = True
                        index = i
                        break
                    elif (name_1 == n3) and (name_3 == n1):
                        found = True
                        index = i
                        break
            if found and (arg.replace is True):
                angles[index] = arg
            elif found and (arg.new is True):
                error_str = (f'Cannot add angle type {repr(arg)}, an angle '
                             f' between {name_1}, {name_2}, and {name_3} '
                             f'already exists')
                raise ValueError(error_str)
            elif (not found) and (arg.new is True):
                angles.append(arg)
            elif (not found) and (arg.replace is True):
                error_str = (f'No existing angle between {name_1}, {name_2}, '
                             f'and {name_3} was found to replace with '
                             f'{repr(arg)}')
                raise ValueError(error_str)
    return angles


def replace_in_fort3(input_file, output_path, *args):
    """Replace or add to an existing fort.3 file

    The output file is named <input_file_name>_new by default, if no other
    output name is supplied.

    The replacement specification is done using Fort3Replacement instances,
    any number of which may be given as positional arguments
    """
    if (output_path is None) or (output_path == ''):
        output_path = os.path.join(os.path.dirname(input_file),
                                   input_file + '_new')
    for arg in args:
        if (not arg.new) and (not arg.replace):
            error_str = (f'Property {repr(arg)} has neither the new or the'
                         ' replace flag set')
            raise ValueError(error_str)

    atom_name, atoms, bonds, angles, torsions, non_bonds, scf, kappa, chi = (
        _parse_fort_3_file(input_file)
    )
    atom_name, atoms = _sort_new_replace_args_atom(atom_name, atoms, *args)
    bonds = _sort_new_replace_args_bonds(atom_name, bonds, *args)

    # new_counts = _count_property_instances(*args)
    # existing_counts = _count_existing_instances(input_file)
    # total = {key: new_counts[key] + existing_counts[key]
    #          for key in new_counts}

    return output_path
