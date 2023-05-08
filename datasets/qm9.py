default_qm9_dir = '~/datasets/molecular/qm9/'

def QM9datasets(root_dir=default_qm9_dir):
    root_dir = os.path.expanduser(root_dir)
    filename= f"{root_dir}data.pz"
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1),
         "data", 'qm9', subtract_thermo=True,force_download=True)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)
            dataset.num_species = 5
            dataset.charge_scale = 9
        os.makedirs(root_dir, exist_ok=True)
        torch.save((datasets, num_species, charge_scale),filename)
        return (datasets, num_species, charge_scale)

