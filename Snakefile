rule all:
  input: 
    expand('Report/{name}_Corr.pdf', name=config['name']),
    expand('Report/{name}_Corr_Conc.pdf', name=config['name']),
    expand('Report/{name}_linear_model.pdf', name=config['name']),
    expand('Report/{name}_CorrHeatmap.pdf', name=config['name']),
    expand('Report/{name}_Corr_JustLambda.pdf', name=config['name']),
    expand('Report/{name}_inv_comp.pdf', name=config['name']),
    expand('Report/{name}_rejected_EOS.pdf', name=config['name']),
    expand('Report/{name}_accepted_EOS.pdf', name=config['name'])

rule generate:
  input: 'MakeSkyrmeFileBisection.py'
  output: 'Results/{name}.Gen.h5'
  shell:
    '''
    module load GNU/8.2.0-2.31.1
    module load OpenMPI/4.0.0
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
