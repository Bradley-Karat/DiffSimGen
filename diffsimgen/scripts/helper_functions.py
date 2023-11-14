from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

def get_acq_scheme(bval,bvec,delta=None,Delta=None,TE=None):
    
    if type(bval) == str:
        bval = np.loadtxt(f'{bval}')
        bvec = np.loadtxt(f'{bvec}')

    if max(bval) < 1e8:
        print('Converting bvalues to s/m^2')
        bval = bval * 1e6

    acq_scheme = acquisition_scheme_from_bvalues(bval, bvec.T, delta, Delta, TE)

    return(acq_scheme)