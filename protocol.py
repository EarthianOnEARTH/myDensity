##### This is a protocol file #####



def protocol(filename = 'toto', filelength = 2):
    
    import MyDensity
    md = MyDensity.Density()

    #Please define all parameters below
    
    
    Experimental_directory = '/pathway/to/your/data/'
    Sample_radius = 1.8
    Capsule_radius = 2.4
    
    ### Energy parameters (in keV) ###
    
    Energy = 30.8
    delta_Energy = 0.5
    Tomo_filepath = 'pathway/to/your/tomo/data'
    
    ### Estimated Value of mu_pho for sample and capsule 
    mu_pho_sample = 3.3
    mu_pho_capsule = 2.1
    
    
    md.setExpDir(Experimental_directory)
    md.setParameters(filename = filename, rsam = Sample_radius, renv = Capsule_radius, energy = Energy, delta_E = delta_Energy, mu_pho_abs = mu_pho_sample, mu_pho_env = mu_pho_capsule, allChannel = True)
    md.loadData()   
    
    
    #Please select all function to process by commenting/uncommenting them
    
    #md.crop(14,0)
    """ This function crops the data in case of bad profile outside the sample region. Arguments are number 
    of points to remove on the left and on the right
    """
    #md.correctBaseline(3)
    """ This function correct the baseline with a linear function based on the number of point on the left and on the right of the absorption profile
    """
    md.function_fit(normalization = 'mean', Norm_point = 1, double_capsule = False)
    """ This function fits the data considering a double capsule to take into account hBN/graphite contribution"""
    md.saveData()




    Density = md.mu_pho_abs
    Density_env = md.mu_pho_env
    sampleRadius = 10000*md.r_sam
    Density_err = 100*md.dif.std()
    capsuleRadius = md.r_env
    
    return Density, Density_env, Density_err, sampleRadius, capsuleRadius
