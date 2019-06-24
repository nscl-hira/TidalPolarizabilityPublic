localrules: add_weight, draw_correlation, draw_corrheatmap, find_coef, draw_coef, all, draw_correlation_concentrated, draw_corrjustlambda

rule all:
  input: 
    expand('Report/{name}.Corr.pdf', name=config['name']),
    expand('Report/{name}.Corr.Conc.pdf', name=config['name']),
    expand('Report/{name}_mass_1.2.pdf', name=config['name']),
    expand('Report/{name}.CorrHeatmap.pdf', name=config['name']),
    expand('Report/{name}.Corr.JustLambda.pdf', name=config['name'])

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
  output: 'Report/{name}.Corr.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationMatrix {output} {input.data}
    '''

rule draw_correlation_concentrated:
  input: 
    wc = 'Plots/DrawCorrelationMatrixConcentrated.py',
    data = 'Results/{name}.Gen.h5', 
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}.Corr.Conc.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationMatrixConcentrated {output} {input.data}
    '''

rule draw_corrheatmap:
  input:
    wc = 'Plots/DrawCorrelationHeatMap.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}.CorrHeatmap.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationHeatMap {output} {input.data}
    '''

rule draw_corrjustlambda:
  input:
    wc = 'Plots/DrawCorrelationJustLambda.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5'
  output: 'Report/{name}.Corr.JustLambda.pdf'
  shell:
    '''
    python -m Plots.DrawCorrelationJustLambda {output} {input.data}
    '''

rule find_coef:
  input: 
    wc = 'Plots/FindLinearCoefficient.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
  output: 'Results/{name}.Gen.Coef.csv'
  shell:
    '''
    python -m Plots.FindLinearCoefficient {output} {input.data}
    '''

rule draw_coef:
  input:
    wc = 'Plots/PlotLinearCoefficient.py',
    data = 'Results/{name}.Gen.h5',
    weight = 'Results/{name}.Gen.Weight.h5',
    coef = 'Results/{name}.Gen.Coef.csv'
  output: 'Report/{name}_mass_1.2.pdf', 'Report/{name}_mass_1.4.pdf', 'Report/{name}_mass_1.6.pdf'
  shell:
    '''
    python -m Plots.PlotLinearCoefficient {input.coef} {input.data} 'Report/{wildcards.name}'
    '''
