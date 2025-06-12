@testset "Experimental_Sparsity_Pattern" begin
    # Things for model
    n = 3
    k = 4
    
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)

    sparsity_1 = VecchiaMLE.SparsityPattern(ptGrid, k, "CSC")
    sparsity_2 = VecchiaMLE.SparsityPattern(ptGrid, k, "Experimental")
    printing = ["rows", "cols", "colptr"]
    @test length(sparsity_1) == length(sparsity_2) # check if rows, cols, colptr are there
    for i in eachindex(sparsity_1)
        println(printing[i])
        println("\tCSC")
        println("\t$(sparsity_1[i])")
        println("\tExperimental")
        println("\t$(sparsity_2[i])")
        @test isequal(sparsity_1[i], sparsity_2[i])
    end
end