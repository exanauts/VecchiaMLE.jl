function create_config(config_file_name::String)::ConfigManager
    config::Dict = TOML.parsefile(config_file_name) 

    # Sanity checks
    if length(config) == 0
        @Error "$config_file_name has no entries."
    end

    if !haskey(config, "General")
        @Error "$config_file_name must have at least the \"General\" section."
    end
    if !haskey(config["General"], "n") || !haskey(config["General"], "k") || !haskey(config["General"], "mode")
        @Error "$config_file_name must have at least entries n, k, and mode."
    end
    
    # parse file and init defaults
    # General Stuff
    n = config["General"]["n"]
    k = config["General"]["k"]
    mode = config["General"]["mode"]

    # Optionals    

    # Samples 
    if haskey(config["Flags"], "samples") && config["Flags"]["samples"] != ""
        samples = read_matrix_from_file(config["Optionals"]["samples"])
    else
        # Get Number of Samples
        if !haskey(config["Optionals"], "Number_of_Samples") || config["Optionals"]["Number_of_Samples"] < 1
            Number_of_Samples = 100        
        else
            Number_of_Samples = config["Optionals"]["Number_of_Samples"]
        end
        # Get Params
        if !haskey(config["Optionals"], "params")
            params = [5.0, 0.2, 2.25, 0.25]
        else
            params = config["Optionals"]["params"]
        end


        MatCov = generate_MatCov(n, params)
        samples = generate_Samples(MatCov, n, Number_of_Samples)
    end
    

    if !haskey(config["Flags"], "MadNLP_Print_Level") || !config["Flags"]["MadNLP_Print_Level"] in 1:5
        MadNLP_Print_Level = 1
    else
        MadNLP_Print_Level = config["Flags"]["MadNLP_Print_Level"]
    end

    return ConfigManager(
        n, 
        k, 
        COMPUTE_MODE(mode),
        Number_of_Samples,
        MadNLP_Print_Level,
        samples
    )
end

