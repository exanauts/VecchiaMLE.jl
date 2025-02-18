var documenterSearchIndex = {"docs":
[{"location":"api/#VecchiaMLE-Defines","page":"API","title":"VecchiaMLE Defines","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"VecchiaMLE.COMPUTE_MODE\nVecchiaMLE.PRINT_LEVEL\nVecchiaMLE.ConfigManager\nVecchiaMLE.VecchiaCache\nVecchiaMLE.Diagnostics\nVecchiaMLE.VecchiaModel\nVecchiaMLE.VecchiaMLEInput","category":"page"},{"location":"api/#VecchiaMLE.COMPUTE_MODE","page":"API","title":"VecchiaMLE.COMPUTE_MODE","text":"Computation mode for which the analysis to run. Generally, we should see better performance at higher n values for the GPU.\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.PRINT_LEVEL","page":"API","title":"VecchiaMLE.PRINT_LEVEL","text":"Print level of the program. Not implemented yet, but will be given by the user to determine the print level of both VecchiaMLE and MadNLP.\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.ConfigManager","page":"API","title":"VecchiaMLE.ConfigManager","text":"At the moment, not used!\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.VecchiaCache","page":"API","title":"VecchiaMLE.VecchiaCache","text":"Internal struct from which to fetch persisting objects in the optimization function. There is no need for the user to mess with this!\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.Diagnostics","page":"API","title":"VecchiaMLE.Diagnostics","text":"The Diagnostics struct that records some internals of the program that would be otherwise difficult to recover. The fields to the struct are as follows:\n\ncreate_model_time: Time taken to create Vecchia Cache and MadNLP init.              \nLinAlg_solve_time: Time in LinearAlgebra routines in MadNLP\nsolve_model_time: Time taken to solve model in MadNLP\nobjective_value: Optimal Objective value. \nnormed_constraint_value: Optimal norm of constraint vector.\nnormed_grad_value: Optimal norm of gradient vector.\nMadNLP_iterations: Iterations for MadNLP to reach optimal.\nmode: Operation mode: CPU or GPU\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.VecchiaModel","page":"API","title":"VecchiaMLE.VecchiaModel","text":"Struct needed for NLPModels.  There is no need for the user to mess with this!\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE.VecchiaMLEInput","page":"API","title":"VecchiaMLE.VecchiaMLEInput","text":"Input to the VecchiaMLE analysis, needs to be filled out by the user! The fields to the struct are as follows:\n\nn: The Square root size of the problem. I.e., the length of one side of ptGrid.  \nk: The \"number of neighbors\", number of conditioning points regarding the Vecchia Approximation. \nsamples: The Samples in which to generate the output. If you have no samples consult the Documentation.\nNumber_of_Samples: The Number of Samples the user gives input to the program.\nMadNLP_print_level: The print level to the optimizer. Can be ignored, and will default to ERROR.\nmode: The opertaing mode to the analysis(GPU or CPU). The mapping is [1: 'CPU', 2: 'GPU'].\n\n\n\n\n\n","category":"type"},{"location":"api/#VecchiaMLE-input","page":"API","title":"VecchiaMLE input","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"VecchiaMLE.VecchiaMLE_Run\nVecchiaMLE.VecchiaMLE_Run_Analysis!\nVecchiaMLE.ExecuteModel!","category":"page"},{"location":"api/#VecchiaMLE.VecchiaMLE_Run","page":"API","title":"VecchiaMLE.VecchiaMLE_Run","text":"diagnostics, result = VecchiaMLE_Run(iVecchiaMLE::VecchiaMLEInput)\n\nMain function to run the analysis. Make sure that your VecchiaMLEInput struct is \nfilled out correctly!\n\nInput arguments\n\niVecchiaMLE: The filled out VecchiaMLEInput struct. See VecchiaMLEInput struct for more details.\n\nOutput arguments\n\ndiagnostics: Some diagnostics of the analysis that would be difficult otherwise to get. See Diagnostics struct for more details.\npres_chol: The approximate precision's cholesky, generated by the samples. \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.VecchiaMLE_Run_Analysis!","page":"API","title":"VecchiaMLE.VecchiaMLE_Run_Analysis!","text":"See VecchiaMLE_Run().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.ExecuteModel!","page":"API","title":"VecchiaMLE.ExecuteModel!","text":"See VecchiaMLE_Run().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE-utils","page":"API","title":"VecchiaMLE utils","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"VecchiaMLE.covariance2D\nVecchiaMLE.generate_Samples\nVecchiaMLE.generate_MatCov\nVecchiaMLE.generate_xyGrid\nVecchiaMLE.generate_safe_xyGrid\nVecchiaMLE.MadNLP_Print_Level\nVecchiaMLE.Int_to_Mode\nVecchiaMLE.Uni_Error\nVecchiaMLE.IndexReorder\nVecchiaMLE.IndexMaxMin\nVecchiaMLE.dist_from_set\nVecchiaMLE.IndexRandom\nVecchiaMLE.KLDivergence\nVecchiaMLE.SparsityPattern\nVecchiaMLE.SparsityPattern_Block\nVecchiaMLE.SparsityPattern_CSC\nVecchiaMLE.sanitize_input!","category":"page"},{"location":"api/#VecchiaMLE.covariance2D","page":"API","title":"VecchiaMLE.covariance2D","text":"Covariance_Matrix = covariance2D(xyGrid::AbstractVector, \n                                 params::AbstractVector)\n\nGenerate a Matern-Like Covariance Matrix for the parameters and locations given.\nNote: This should not be called by the user. This is the back end for the function\ngenerate_MatCov()!\n\nInput arguments\n\nxyGrid: A set of points in 2D space upon which we determine the indices of the Covariance matrix;\nparams: An array of length 3 (or 4) that holds the parameters to the matern covariance kernel (σ, ρ, ν).\n\nOutput arguments\n\nCovariance_Matrix : An n × n Matern matrix, where n is the length of the xyGrid \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.generate_Samples","page":"API","title":"VecchiaMLE.generate_Samples","text":"Samples_Matrix = generate_Samples(MatCov::AbstractMatrix, \n                                  n::Integer,\n                                  Number_of_Samples::Integer)\n\nGenerate a number of samples according to the given Covariance Matrix MatCov.\nNote the samples are given as mean zero. \nIf a CUDA compatible device is detected, the samples are generated on the GPU\nand transferred back to the CPU.\n\nInput arguments\n\nMatCov: A Covariance Matrix, presumably positive definite;\nn: The length of one side of the Covariance matrix;\nNumber_of_Samples: How many samples to return.\n\nOutput arguments\n\nSamples_Matrix : A matrix of size (NumberofSamples × n), where the rows are the i.i.d samples  \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.generate_MatCov","page":"API","title":"VecchiaMLE.generate_MatCov","text":"Covariance_Matrix = generate_MatCov(n::Integer,\n                                    params::AbstractArray,\n                                    ptGrid::AbstractVector)\n\nGenerates a Matern Covariance Matrix determined via the given paramters (params) and ptGrid.\n\nInput arguments\n\nn: The length of the ptGrid;\nparams: An array of length 3 (or 4) that holds the parameters to the matern covariance kernel (σ, ρ, ν);\nptGrid: A set of points in 2D space upon which we determine the indices of the Covariance matrix;\n\nOutput arguments\n\nCovariance_Matrix : An n × n Symmetric Matern matrix \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.generate_xyGrid","page":"API","title":"VecchiaMLE.generate_xyGrid","text":"xyGrid = generate_xyGrid(n::Integer)\n\nA helper function to generate a point grid which partitions the positive unit square [0, 1] × [0, 1].\nNOTE: This function at larger n values (n > 100) causes ill conditioning of the genreated Covariance matrix!\nSee generate_safe_xyGrid() for higher n values.\n\nInput arguments\n\nn: The length of the desired xyGrid;\n\nOutput arguments\n\nxyGrid : The desired points in 2D space  \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.generate_safe_xyGrid","page":"API","title":"VecchiaMLE.generate_safe_xyGrid","text":"xyGrid = generate_safe_xyGrid(n::Integer)\n\nA helper function to generate a point grid which partitions the positive square [0, k] × [0, k], \nwhere k = cld(n, 10). This should avoid ill conditioning of a Covariance Matrix generated by these points.\n\nInput arguments\n\nn: The length of the desired xyGrid;\n\nOutput arguments\n\nxyGrid : The desired points in 2D space  \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.MadNLP_Print_Level","page":"API","title":"VecchiaMLE.MadNLP_Print_Level","text":"log_level = MadNLP_Print_Level(pLEvel::Integer)\n\nA helper function to convert an integer to a MadNLP LogLevel.\nThe mapping is [1: 'TRACE', 2: 'DEBUG', 3: 'INFO', 4: 'WARN', 5: 'ERROR'].\nAny other integer given is converted to MadNLP.Fatal.\n\nInput arguments\n\npLevel: The given log level as an integer;\n\nOutput arguments\n\nlog_level : The coded MadNLP.LogLevel.   \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.Int_to_Mode","page":"API","title":"VecchiaMLE.Int_to_Mode","text":"cpu_mode = Int_to_Mode(n::Integer)\n\nA helper function to convert an integer to a COMPUTE_MODE.\nThe mapping is [1: 'CPU', 2: 'GPU'].\nAny other integer given is converted CPU.\n\nInput arguments\n\nn: The given cpu mode as an integer;\n\nOutput arguments\n\ncpu_mode : The coded COMPUTE_MODE.   \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.Uni_Error","page":"API","title":"VecchiaMLE.Uni_Error","text":"Error = Uni_Error(TCov::AbstractMatrix,\n                  L::AbstractMatrix)\nGenerates the \"Univariate KL Divergence\" from the Given True Covariane matrix \nand Approximate Covariance matrix. The output is the result of the following \noptimization problem: sup(a^T X_A || a^⊺ X_T). The solution is \n            f(μ) = ln(μ) + (2μ)^{-2} - 0.5,\nwhere μ is the largest or smallest eigenvalue of the matrix Σ_A^{-1}Σ_T, whichever\nmaximizes the function f(μ). Note: Σ_A , Σ_T are the respective approximate and true\ncovariance matrices, and X_A, X_T are samples from the respective Distributions (mean zero).\n\nInput arguments\n\nTCov: The True Covariance Matrix: the one we are approximating;\nL: The cholesky factor of the approximate precision matrix.\n\nOutput arguments\n\nf(μ) : the result of the function f(μ) detailed above.   \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.IndexReorder","page":"API","title":"VecchiaMLE.IndexReorder","text":"indx_perm, dist_set = IndexReorder(Cond_set::AbstractVector,\n                                   data::AbstractVector,\n                                   mean_mu::AbstractVector,\n                                   method::String = \"random\", \n                                   reverse_ordering::Bool = true)\n\nFront End function to determine the reordering of the given indices (data).\nAt the moment, only random, and maxmin are implemented.    \n\nI BELIEVE the intention is to permute the coordinates of the samples with it, e.g.,\nsamples = samples[:, indx_perm]. Note for small conditioning sets per point (extremely sparse L),\nthis is not worth while!.\n\nInput arguments\n\nCond_set: A set of points from which to neglect from the index permutation;\ndata: The data to determine said permutation;\nmean_mu: The theoretical mean of the data, not mean(data)!\nmethod: Either permuting based on maxmin, or random.\nreverse_ordering: permutation is generated either in reverse or not.   \n\nOutput arguments\n\nindx_perm: The index set which permutes the given data.\ndist_set: The array of maximum distances to iterating conditioning sets.    \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.IndexMaxMin","page":"API","title":"VecchiaMLE.IndexMaxMin","text":"See IndexReorder().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.dist_from_set","page":"API","title":"VecchiaMLE.dist_from_set","text":"Helper function. See IndexReorder().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.IndexRandom","page":"API","title":"VecchiaMLE.IndexRandom","text":"See IndexReorder().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.KLDivergence","page":"API","title":"VecchiaMLE.KLDivergence","text":"KL_Divergence = KLDivergence(TCov::Symmetric{Float64},\n                             AL::AbstractMatrix)\n\nComputes the KL Divergence of the True Covariance matrix, TCov, and\nThe APPROXIMATE INVERSE CHOLESKY FACTOR, AL. Assumed mean zero.\nNote: This is extremely slow!\n\nInput arguments\n\nTCov: The True Covariance Matrix;\nAL: The cholesky factor to the approximate precision matrix. I.e., The ouptut of VecchiaMLE.  \n\nOutput arguments\n\nres: The result of the KL Divergence function.\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.SparsityPattern","page":"API","title":"VecchiaMLE.SparsityPattern","text":"rows, cols, colptr = SparsityPattern(data::AbstractVector,\n                                     k::Integer,\n                                     format::String = \"\")\n\nFront end to generate the sparsity pattern of the approximate \nprecision's cholesky factor, L, in CSC format. The pattern is \ndetermined by nearest neighbors of the previous points in data.\n\nInput arguments\n\ndata: The point grid that was used to either generate the covariance matrix, or any custom ptGrid.\nk: The number of nearest neighbors for each point (Including the point itself)\nformat: Either blank or \"CSC\". This can be ignored.\n\nOutput arguments\n\nrows: A vector of row indices of the sparsity pattern for L, in CSC format.\ncols: A vector of column indices of the sparsity pattern for L, in CSC format.\ncolptr: A vector of incides which determine where new columns start. \n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.SparsityPattern_Block","page":"API","title":"VecchiaMLE.SparsityPattern_Block","text":"See SparsityPattern().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.SparsityPattern_CSC","page":"API","title":"VecchiaMLE.SparsityPattern_CSC","text":"See SparsityPattern().\n\n\n\n\n\n","category":"function"},{"location":"api/#VecchiaMLE.sanitize_input!","page":"API","title":"VecchiaMLE.sanitize_input!","text":"sanitize_input!(iVecchiaMLE::VecchiaMLEInput)\n\nA helper function to catch any inconsistencies in the input given by the user.\n\nThe current checks are:\n\n* Ensuring n > 0.\n* Ensuring k <= n^2 (Makes sense considering the ptGrid and SparsityPattern sizes).\n* Ensuring the sample matrix, if the user gives one, is nonempty and is the same size as n^2.\n* Ensuring the MadNLP_print_level in 1:5. See MadNLP_Print_Level().\n* Ensuring the mode in 1:2. See Int_to_Mode().\n\nInput arguments\n\niVecchiaMLE: The filled-out VecchiaMLEInput struct. See VecchiaMLEInput struct for more details. \n\n\n\n\n\n","category":"function"},{"location":"#Home","page":"Home","title":"VecchiaMLE.jl documentation","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package computes an approximate Cholesky factorization of a covariance matrix using a Maximum Likelihood Estimation (MLE) approach. The Cholesky factor is computed via the Vecchia approximation, which is sparse and approximately banded, making it highly efficient for large-scale problems.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> ]\npkg> add https://github.com/exanauts/VecchiaMLE.jl.git\npkg> test VecchiaMLE","category":"page"},{"location":"examples/HowToRun/#How-To-Run-An-Analysis","page":"Tutorials","title":"How To Run An Analysis","text":"","category":"section"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"This example shows how to run VecchiaMLE for a given VecchiaMLEInput. There are  some necessary parameters the user needs to input, which are the following:","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"using VecchiaMLE\nusing SparseArrays # To show matrix at the end \n\n# Things for model\nn = 20\nk = 10\nNumber_of_Samples = 100","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"These are the three major parameters: n (dimension), k (conditioning for Vecchia),  and the NumberofSamples. they should be self-explanatory, but the  documentation for the VecchiaMLEInput should clear things up. Next, we should  generate the samples. At the moment, the code has only been verified to run for generated samples by the program, but there should be no difficulty in inputting your own samples. Just make sure the length of each sample should be a perfect square. This is a strange restriction, and will be removed when we allow user input locations. Nevertheless, let's move on. ","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"The sample generation requires a covariance matrix. Here, we will generate a matern like covairance matrix which VecchiaMLE has a predefined function for. Although not exposed directly to the user, we allow the generation of samples in the following manner:","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"params = [5.0, 0.2, 2.25, 0.25]\nMatCov = VecchiaMLE.generate_MatCov(n, params)\nsamples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"Next, we create and fill in the VecchiaMLEInput struct. This is done below.","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1)","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"Here, we set (though not necessary) the optimizer (MadNLP) print level to 5 (ERROR),  which keep it silent. Furthermore, we set the compute mode to 1 (CPU). The other option  would be to set it to 2 (GPU), but since not all machines have a GPU connected, we opt for the sure-fire approach.","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"All that's left is to run the analysis. This is done in one line:","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"d, o = VecchiaMLE_Run(input)","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"We can see the structure of the output of the program has an approximately banded structure.  This is due to the generation of the sparsity pattern depends on the euclidean distance between any given point and previously considered points in the grid.","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"display(sparse(o))","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"The function VecchiaMLE_Run also returns some diagnostics that would be difficult otherwise to retrieve, and the result of the analysis. This is the cholesky factor to the approximate precision matrix. You can check the KL Divergence of this approximation, with a function inside VecchiaMLE,  though this is not recommended for larger dimensions (n >= 30). This obviously requires the true covariance matrix, which should be generated here if not before. ","category":"page"},{"location":"examples/HowToRun/","page":"Tutorials","title":"Tutorials","text":"println(\"Error: \", VecchiaMLE.KLDivergence(MatCov, o))","category":"page"}]
}
