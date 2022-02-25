rule all:
  input: 
    #expand('Report/{name}_Corr.pdf', name=config['name']),
    #expand('Report/{name}_Corr_Conc.pdf', name=config['name']),
    #expand('Report/{name}_linear_model.pdf', name=config['name']),
    #expand('Report/{name}_CorrHeatmap.pdf', name=config['name']),
    #expand('Report/{name}_Corr_JustLambda.pdf', name=config['name']),
    #expand('Report/{name}_inv_comp.pdf', name=config['name']),
    #expand('Report/{name}_rejected_EOS.pdf', name=config['name']),
    #expand('Report/{name}_accepted_EOS.pdf', name=config['name']),
    expand('Report/{name}_SymPressure.pdf', name=config['name']),
    expand('Report/{name}_PostPressure.pdf', name=config['name']),
    expand('Report/{name}_AsymPost.pdf', name=config['name']),
    expand('Report/{name}_Post.pdf', name=config['name']),



rule generate:
  input: ancient('MakeSkyrmeFileBisection.py')
  output: 'Results/{name}.Gen.h5'
  shell:
    '''
    #module load GNU/8.2.0-2.31.1
    #module load OpenMPI/4.0.0
    export OMPI_MCA_btl_openib_allow_ib=1
    mpirun python -W ignore {input} -o {wildcards.name} --enable-debug --Gen
    ''' 

rule add_weight:
  input: 
    wc = 'AddWeight.py',
    data = 'Results/{name}.Gen.h5'
  output: 'Results/{name}.Gen.Weight.h5'
  shell:
    '''
    python {input.wc} {input.data}
    '''

rule draw_SymPressure:
  input:
    wc = 'Plots/DrawAcceptedEOSSymPressure2.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 
    pdf = 'Report/{name}_PostPressure.pdf',
    pkl = 'Report/{name}_PostPressure.pkl'
  shell:
    '''
    #module load GNU/8.2.0-2.31.1
    #module load OpenMPI/4.0.0
    export OMPI_MCA_btl_openib_allow_ib=1
    mpirun python -m Plots.DrawAcceptedEOSSymPressure2 {output.pdf} {input.data}
    python -m Plots.OverlayNICER {output.pkl} {output.pdf}
    '''

rule draw_Pressure:
  input:
    wc = 'Plots/DrawAcceptedEOSSpiRIT2.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 
    pdf = 'Report/{name}_SymPressure.pdf',
    pkl = 'Report/{name}_SymPressure.pkl'
  shell:
    '''
    #source /mnt/home/tsangchu/.bashrc_temp
    #export OMPI_MCA_btl_openib_allow_ib=1
    #conda activate Tidal3
    mpirun python -m Plots.DrawAcceptedEOSSpiRIT2 {output.pdf} {input.data}
    python -m Plots.OverlayNICER {output.pkl} {output.pdf}
    '''

rule draw_AsymPost:
  input:
    wc = 'Plots/DrawAcceptedEOSSymTerm.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 
    pdf = 'Report/{name}_AsymPost.pdf',
    pkl = 'Report/{name}_AsymPost.pkl'
  shell:
    '''
    source /mnt/home/tsangchu/.bashrc_temp
    export OMPI_MCA_btl_openib_allow_ib=1
    conda activate Tidal3
    mpirun python -m Plots.DrawAcceptedEOSSymTerm {output.pdf} {input.data}
    python -m OverlayAsym {output.pkl} {output.pdf}
    '''



rule draw_Post:
  input:
    wc = 'Plots/DrawSym15.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 
    pdf = 'Report/{name}_Post.pdf',
    pkl = 'Report/{name}_Post.pkl'
  shell:
    '''
    source /mnt/home/tsangchu/.bashrc_temp
    export OMPI_MCA_btl_openib_allow_ib=1
    conda activate Tidal3
    python -m Plots.DrawSym15 {output.pdf} {input.data}
    '''



rule draw_correlation:
  input: 
    wc = 'Plots/DrawCorrelationMatrix.py',
    data = 'Results/{name}.Gen.h5', 
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}_Corr.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationMatrix {output} {input.data}
    '''

rule draw_correlation_concentrated:
  input: 
    wc = 'Plots/DrawCorrelationMatrixConcentrated.py',
    data = 'Results/{name}.Gen.h5', 
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}_Corr_Conc.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationMatrixConcentrated {output} {input.data}
    '''

rule draw_corrheatmap:
  input:
    wc = 'Plots/DrawCorrelationHeatMap.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}_CorrHeatmap.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationHeatMap {output} {input.data}
    '''

rule draw_corrjustlambda:
  input:
    wc = 'Plots/DrawCorrelationJustLambda.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}_Corr_JustLambda.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationJustLambda {output} {input.data}
    '''

rule find_coef:
  input: 
    wc = 'Plots/FindLinearCoefficient.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
  output: 'Results/{name}.Gen_Coef.csv'
  shell:
    '''
    python -m Plots.FindLinearCoefficient {output} {input.data}
    '''

rule draw_coef:
  input:
    wc = 'Plots/PlotLinearCoefficient.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
    coef = 'Results/{name}.Gen_Coef.csv'
  output: 'Report/{name}_linear_model.pdf'
  shell:
    '''
    python -m Plots.PlotLinearCoefficient {input.coef} {input.data} 'Report/{wildcards.name}'
    '''

rule draw_invcomp:
  input:
    wc = 'Plots/PlotInvCompactness.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
  output: 'Report/{name}_inv_comp.pdf'
  shell:
    '''
    python -m Plots.PlotInvCompactness {output} {input.data} 
    '''

rule draw_rejectEOS:
  input:
    wc = 'Plots/DrawRejectedEOS.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
  output: 'Report/{name}_rejected_EOS.pdf'
  shell:
    '''
    python -m Plots.DrawRejectedEOS {output} {input.data} 
    '''

rule draw_acceptEOS:
  input:
    wc = 'Plots/DrawAcceptedEOS.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
  output: 'Report/{name}_accepted_EOS.pdf'
  shell:
    '''
    python -m Plots.DrawAcceptedEOS {output} {input.data} 
    '''
