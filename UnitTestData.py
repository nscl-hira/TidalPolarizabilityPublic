import numpy as np

PRCTransDensity = 0.3
PolyTropeDensity = 3*0.16
MaxMass = 2


density = np.logspace(np.log(1e-4), np.log(10*0.16), 100, base=np.e)
EOSTransKwargs = {'PRCTransDensity': 0.3, 'PolyTropeDensity': 3*0.16,
                  'MaxMass': 2.}
EOSTransDens = np.array([0.024697700269418633, 0.08232566756472878, 0.48]) 
EOSPressure = np.array([1.57136877e-04, 1.78572198e-04, 2.03601126e-04, 2.30176426e-04,
                        2.58547012e-04, 2.90334612e-04, 3.26683290e-04, 3.66552521e-04,
                        4.07539733e-04, 4.45303764e-04, 4.75422586e-04, 4.97457731e-04,
                        5.12813842e-04, 5.25107567e-04, 5.41381706e-04, 5.72169720e-04,
                        6.21187806e-04, 6.86335068e-04, 7.61946412e-04, 8.37843181e-04,
                        9.08816201e-04, 9.75907124e-04, 1.04222788e-03, 1.11327214e-03,
                        1.19201767e-03, 1.27949163e-03, 1.37689917e-03, 1.48569399e-03,
                        1.60765059e-03, 1.74494633e-03, 1.90012752e-03, 2.07622350e-03,
                        2.27695863e-03, 2.50695123e-03, 2.77197414e-03, 3.07929836e-03,
                        3.43814726e-03, 3.86029807e-03, 4.36003674e-03, 4.95146305e-03,
                        5.64981069e-03, 6.47214531e-03, 7.43736385e-03, 8.56801877e-03,
                        9.89136181e-03, 1.14383769e-02, 1.32438271e-02, 1.53476597e-02,
                        1.77995149e-02, 2.06576181e-02, 2.39889796e-02, 2.78702027e-02,
                        3.23881174e-02, 3.76400830e-02, 4.37337310e-02, 5.07875071e-02,
                        5.89555269e-02, 6.93128672e-02, 8.16366245e-02, 9.56777062e-02,
                        1.11675298e-01, 1.29901894e-01, 1.50667938e-01, 1.74327104e-01,
                        2.01282319e-01, 2.31992623e-01, 2.66980978e-01, 3.06843176e-01,
                        3.52257978e-01, 4.16878085e-01, 5.24735800e-01, 6.62729487e-01,
                        8.40066688e-01, 1.06903917e+00, 1.36614655e+00, 1.75334299e+00,
                        2.25895197e+00, 2.92182545e+00, 3.79460976e+00, 4.94857046e+00,
                        6.45315779e+00, 8.38817822e+00, 1.08619271e+01, 1.40014432e+01,
                        1.79502480e+01, 2.28746429e+01, 2.90832839e+01, 3.81423509e+01,
                        5.37299975e+01, 7.56878525e+01, 1.06619231e+02, 1.50191346e+02,
                        2.11570090e+02, 2.98032505e+02, 4.19829543e+02, 5.91401415e+02,
                        8.33089617e+02, 1.17354861e+03, 1.65314309e+03, 2.32873360e+03]) 
EOSEnergyDensity = np.array([9.36423773e-02, 1.03237812e-01, 1.13904766e-01, 1.25648458e-01,
                             1.38527162e-01, 1.52728755e-01, 1.68462482e-01, 1.85841670e-01,
                             2.04949740e-01, 2.25982815e-01, 2.49199616e-01, 2.74839769e-01,
                             3.03156287e-01, 3.34419426e-01, 3.68909378e-01, 4.06910062e-01,
                             4.48756142e-01, 4.94844070e-01, 5.45636182e-01, 6.01683893e-01,
                             6.63576322e-01, 7.31919839e-01, 8.07357947e-01, 8.90561123e-01,
                             9.82287572e-01, 1.08339700e+00, 1.19483905e+00, 1.31766832e+00,
                             1.45306619e+00, 1.60236733e+00, 1.76703091e+00, 1.94864434e+00,
                             2.14895950e+00, 2.36990965e+00, 2.61362716e+00, 2.88246194e+00,
                             3.17900000e+00, 3.50608091e+00, 3.86682018e+00, 4.26466077e+00,
                             4.70339645e+00, 5.18719794e+00, 5.72064720e+00, 6.30880285e+00,
                             6.95727718e+00, 7.67230560e+00, 8.46085638e+00, 9.33070926e+00,
                             1.02903658e+01, 1.13491950e+01, 1.25175499e+01, 1.38068589e+01,
                             1.52297118e+01, 1.67999337e+01, 1.85326277e+01, 2.04441905e+01,
                             2.25527162e+01, 2.48771532e+01, 2.74397998e+01, 3.02656854e+01,
                             3.33818504e+01, 3.68181126e+01, 4.06073529e+01, 4.47858299e+01,
                             4.93935264e+01, 5.44745327e+01, 6.00774677e+01, 6.62559448e+01,
                             7.30690847e+01, 8.05933871e+01, 8.89201107e+01, 9.81147021e+01,
                             1.08269848e+02, 1.19488851e+02, 1.31887087e+02, 1.45593748e+02,
                             1.60753853e+02, 1.77530725e+02, 1.96109119e+02, 2.16699074e+02,
                             2.39539809e+02, 2.64902234e+02, 2.93094906e+02, 3.24470153e+02,
                             3.59430413e+02, 3.98434952e+02, 4.42013809e+02, 4.90804124e+02,
                             5.45880271e+02, 6.08518326e+02, 6.80273330e+02, 7.63178118e+02,
                             8.59922380e+02, 9.74103229e+02, 1.11057634e+03, 1.27594858e+03,
                             1.47926967e+03, 1.73300415e+03, 2.05439776e+03, 2.46739924e+03])
EOSMaxMass = {'PCentral': 1397.802617745039, 'DensCentral': 1.3831226392437435, 'mass': 1.9999999933246129, 'Radius': 9.021997516475832, 'Lambda': 2.183813786059423}
EOS1_4Mass = {'PCentral': 146.94256632477268, 'DensCentral': 0.7272485532984097, 'mass': 1.3999999999999948, 'Radius': 10.43875901956453, 'Lambda': 133.9745925698687}


EOS2PolyTransDens = np.array([0.024697700269418633, 0.08232566756472878, 0.48] )
EOS2PolyPressure = np.array([1.57136877e-04, 1.78572198e-04, 2.03601126e-04, 2.30176426e-04,
                             2.58547012e-04, 2.90334612e-04, 3.26683290e-04, 3.66552521e-04,
                             4.07539733e-04, 4.45303764e-04, 4.75422586e-04, 4.97457731e-04,
                             5.12813842e-04, 5.25107567e-04, 5.41381706e-04, 5.72169720e-04,
                             6.21187806e-04, 6.86335068e-04, 7.61946412e-04, 8.37843181e-04,
                             9.08816201e-04, 9.75907124e-04, 1.04222788e-03, 1.11327214e-03,
                             1.19201767e-03, 1.27949163e-03, 1.37689917e-03, 1.48569399e-03,
                             1.60765059e-03, 1.74494633e-03, 1.90012752e-03, 2.07622350e-03,
                             2.27695863e-03, 2.50695123e-03, 2.77197414e-03, 3.07929836e-03,
                             3.43814726e-03, 3.86029807e-03, 4.36003674e-03, 4.95146305e-03,
                             5.64981069e-03, 6.47214531e-03, 7.43736385e-03, 8.56801877e-03,
                             9.89136181e-03, 1.14383769e-02, 1.32438271e-02, 1.53476597e-02,
                             1.77995149e-02, 2.06576181e-02, 2.39889796e-02, 2.78702027e-02,
                             3.23881174e-02, 3.76400830e-02, 4.37337310e-02, 5.07875071e-02,
                             5.89555269e-02, 6.84381563e-02, 7.94763530e-02, 9.21154618e-02,
                             1.05501763e-01, 1.19754483e-01, 1.35296190e-01, 1.52816073e-01,
                             1.73397078e-01, 1.98695217e-01, 2.31190841e-01, 2.74538876e-01,
                             3.34054745e-01, 4.16878085e-01, 5.24735800e-01, 6.62729487e-01,
                             8.40066688e-01, 1.06903917e+00, 1.36614655e+00, 1.75334299e+00,
                             2.25895197e+00, 2.92182545e+00, 3.79460976e+00, 4.94857046e+00,
                             6.45315779e+00, 8.38817822e+00, 1.08619271e+01, 1.40014432e+01,
                             1.79502480e+01, 2.28746429e+01, 2.90832839e+01, 4.95158161e+01,
                             1.04829469e+02, 1.71762543e+02, 2.52755861e+02, 3.50762978e+02,
                             4.69357885e+02, 6.12865340e+02, 7.86518579e+02, 9.96650151e+02,
                             1.25092285e+03, 1.55860915e+03, 1.93092934e+03, 2.38146068e+03])
EOS2PolyEnergyDensity = np.array([9.36423773e-02, 1.03237812e-01, 1.13904766e-01, 1.25648458e-01,
                                  1.38527162e-01, 1.52728755e-01, 1.68462482e-01, 1.85841670e-01,
                                  2.04949740e-01, 2.25982815e-01, 2.49199616e-01, 2.74839769e-01,
                                  3.03156287e-01, 3.34419426e-01, 3.68909378e-01, 4.06910062e-01,
                                  4.48756142e-01, 4.94844070e-01, 5.45636182e-01, 6.01683893e-01,
                                  6.63576322e-01, 7.31919839e-01, 8.07357947e-01, 8.90561123e-01,
                                  9.82287572e-01, 1.08339700e+00, 1.19483905e+00, 1.31766832e+00,
                                  1.45306619e+00, 1.60236733e+00, 1.76703091e+00, 1.94864434e+00,
                                  2.14895950e+00, 2.36990965e+00, 2.61362716e+00, 2.88246194e+00,
                                  3.17900000e+00, 3.50608091e+00, 3.86682018e+00, 4.26466077e+00,
                                  4.70339645e+00, 5.18719794e+00, 5.72064720e+00, 6.30880285e+00,
                                  6.95727718e+00, 7.67230560e+00, 8.46085638e+00, 9.33070926e+00,
                                  1.02903658e+01, 1.13491950e+01, 1.25175499e+01, 1.38068589e+01,
                                  1.52297118e+01, 1.67999337e+01, 1.85326277e+01, 2.04441905e+01,
                                  2.25527162e+01, 2.48783013e+01, 2.74430526e+01, 3.02712888e+01,
                                  3.33897281e+01, 3.68276958e+01, 4.06174553e+01, 4.47945932e+01,
                                  4.93985312e+01, 5.44732456e+01, 6.00683166e+01, 6.62405033e+01,
                                  7.30561498e+01, 8.05933871e+01, 8.89201107e+01, 9.81147021e+01,
                                  1.08269848e+02, 1.19488851e+02, 1.31887087e+02, 1.45593748e+02,
                                  1.60753853e+02, 1.77530725e+02, 1.96109119e+02, 2.16699074e+02,
                                  2.39539809e+02, 2.64902234e+02, 2.93094906e+02, 3.24470153e+02,
                                  3.59430413e+02, 3.98434952e+02, 4.42013809e+02, 4.90978634e+02,
                                  5.49203532e+02, 6.19659399e+02, 7.04915523e+02, 8.08080910e+02,
                                  9.32917654e+02, 1.08397813e+03, 1.26677102e+03, 1.48796214e+03,
                                  1.75561762e+03, 2.07949793e+03, 2.47141392e+03, 2.94565743e+03])
EOS2PolyMaxMass = {'PCentral': 845.2631333764994, 'DensCentral': 1.010415875145069, 'mass': 2.3627387662477126, 'Radius': 10.492216145280405, 'Lambda': 2.4949418016350386}
EOS2Poly1_4Mass = {'PCentral': 102.58021569292814, 'DensCentral': 0.543773194261109, 'mass': 1.399999999933259, 'Radius': 10.89801775154346, 'Lambda': 198.81517129015495}


MetaKwargs = {'Esat': -16., 'Esym': 32.775, 'Lsym': 69.86666667, 'Ksat':249.1666667,
              'Ksym':-46.33333333,'Qsat': -110.3333333,'Qsym': 362.5,
              'Zsat':3288.166667,'Zsym':-3970.833333,'msat': 0.731666667,'kv':0.41}
MetaTransKwargs = {'PRCTransDensity': 0.3, 'SpeedOfSound': 0.99}
MetaTransDens = np.array([0.021029999999624998, 0.07009999999875, 0.6008814903837036]) 
MetaPressure = np.array([1.57136877e-04, 1.78572198e-04, 2.03601126e-04, 2.30176426e-04,
                         2.58547012e-04, 2.90334612e-04, 3.26683290e-04, 3.66552521e-04,
                         4.07539733e-04, 4.45303764e-04, 4.75422586e-04, 4.97457731e-04,
                         5.12813842e-04, 5.25107567e-04, 5.41381706e-04, 5.72169720e-04,
                         6.21187806e-04, 6.86335068e-04, 7.61946412e-04, 8.37843181e-04,
                         9.08816201e-04, 9.75907124e-04, 1.04222788e-03, 1.11327214e-03,
                         1.19201767e-03, 1.27949163e-03, 1.37689917e-03, 1.48569399e-03,
                         1.60765059e-03, 1.74494633e-03, 1.90012752e-03, 2.07622350e-03,
                         2.27695863e-03, 2.50695123e-03, 2.77197414e-03, 3.07929836e-03,
                         3.43814726e-03, 3.86029807e-03, 4.36003674e-03, 4.95146305e-03,
                         5.64981069e-03, 6.47214531e-03, 7.43736385e-03, 8.56801877e-03,
                         9.89136181e-03, 1.14383769e-02, 1.32438271e-02, 1.53476597e-02,
                         1.77995149e-02, 2.06576181e-02, 2.39889796e-02, 2.78702027e-02,
                         3.23881174e-02, 3.76400830e-02, 4.37337310e-02, 5.07875070e-02,
                         5.89555091e-02, 6.84186076e-02, 7.87997895e-02, 9.00919379e-02,
                         1.02706937e-01, 1.17287691e-01, 1.34813503e-01, 1.56747868e-01,
                         1.85244584e-01, 2.23433892e-01, 2.75818093e-01, 3.48816676e-01,
                         4.47600529e-01, 5.76672724e-01, 7.46081037e-01, 9.69461541e-01,
                         1.26539816e+00, 1.65932915e+00, 2.18622520e+00, 2.89436400e+00,
                         3.85066730e+00, 5.14827034e+00, 6.91728860e+00, 9.34017528e+00,
                         1.26736840e+01, 1.73933932e+01, 2.43728026e+01, 3.47972872e+01,
                         5.04730621e+01, 7.41507600e+01, 1.10008879e+02, 1.64308981e+02,
                         2.38472417e+02, 3.25079037e+02, 4.29257269e+02, 5.55319622e+02,
                         7.07863167e+02, 8.92450654e+02, 1.11581337e+03, 1.38609660e+03,
                         1.71315668e+03, 2.10892037e+03, 2.58781986e+03, 3.16731901e+03]) 
MetaEnergyDensity = np.array([9.36423773e-02, 1.03237812e-01, 1.13904766e-01, 1.25648458e-01,
                              1.38527162e-01, 1.52728755e-01, 1.68462482e-01, 1.85841670e-01,
                              2.04949740e-01, 2.25982815e-01, 2.49199616e-01, 2.74839769e-01,
                              3.03156287e-01, 3.34419426e-01, 3.68909378e-01, 4.06910062e-01,
                              4.48756142e-01, 4.94844070e-01, 5.45636182e-01, 6.01683893e-01,
                              6.63576322e-01, 7.31919839e-01, 8.07357947e-01, 8.90561123e-01,
                              9.82287572e-01, 1.08339700e+00, 1.19483905e+00, 1.31766832e+00,
                              1.45306619e+00, 1.60236733e+00, 1.76703091e+00, 1.94864434e+00,
                              2.14895950e+00, 2.36990965e+00, 2.61362716e+00, 2.88246194e+00,
                              3.17900000e+00, 3.50608091e+00, 3.86682018e+00, 4.26466077e+00,
                              4.70339645e+00, 5.18719794e+00, 5.72064720e+00, 6.30880285e+00,
                              6.95727718e+00, 7.67230560e+00, 8.46085638e+00, 9.33070926e+00,
                              1.02903658e+01, 1.13491950e+01, 1.25175499e+01, 1.38068589e+01,
                              1.52297118e+01, 1.67999337e+01, 1.85326277e+01, 2.04441905e+01,
                              2.25527162e+01, 2.48782823e+01, 2.74422965e+01, 3.02683511e+01,
                              3.33825875e+01, 3.68137835e+01, 4.05937895e+01, 4.47581719e+01,
                              4.93471918e+01, 5.44073261e+01, 5.99936585e+01, 6.61736495e+01,
                              7.30116942e+01, 8.05637700e+01, 8.89068037e+01, 9.81268401e+01,
                              1.08320400e+02, 1.19596154e+02, 1.32077006e+02, 1.45902708e+02,
                              1.61233215e+02, 1.78253038e+02, 1.97176965e+02, 2.18257701e+02,
                              2.41796183e+02, 2.68158744e+02, 2.97820103e+02, 3.31406937e+02,
                              3.69761121e+02, 4.14040916e+02, 4.65873277e+02, 5.27580212e+02,
                              6.02306625e+02, 6.92893078e+02, 8.02554375e+02, 9.35251589e+02,
                              1.09582374e+03, 1.29012636e+03, 1.52524501e+03, 1.80975367e+03,
                              2.15402743e+03, 2.57062080e+03, 3.07472553e+03, 3.68472463e+03])
MetaMaxMass = {'PCentral': 633.0125800801634, 'DensCentral': 0.7709580503120136, 'mass': 2.675067622337589, 'Radius': 12.095486901902671, 'Lambda': 2.755951533019264}
Meta1_4Mass = {'PCentral': 47.92207700750174, 'DensCentral': 0.36416997309366944, 'mass': 1.3999999999026802, 'Radius': 13.072629998206393, 'Lambda': 634.4383555927775}

