# These macros are for use in validate_input, as well as some tests.

macro assert_eq(given_expr, expected_expr)
    quote
        given_val = $(esc(given_expr))
        expected_val = $(esc(expected_expr))
        @assert given_val == expected_val (
            string($(QuoteNode(given_expr)), " must equal ", $(QuoteNode((expected_expr))),
                   ". Expected: ", expected_val,
                   ", Given: ", given_val, ".")
        )
    end
end

macro assert_cond(cond, var_expr, expected_str)
    quote
        val = $(esc(var_expr))
        @assert $(esc(cond)) (
            string($(QuoteNode(var_expr)), " must ", $(esc(expected_str)),
                   ". Given: ", val, ".")
        )
    end
end

macro assert_cond_compare(expr)
    # expr is like :(iVecchiaMLE.k <= iVecchiaMLE.n)
    if !(expr.head == :call && length(expr.args) == 3)
        throw(ArgumentError("@assert_cond_compare expects a binary operator expression, e.g., a > b"))
    end

    op = expr.args[1]
    given_expr = expr.args[2]
    expected_expr = expr.args[3]

    return quote
        val = $(esc(given_expr))
        expected_val = $(esc(expected_expr))
        op_func = $(esc(op))
        @assert op_func(val, expected_val) (
            string($(QuoteNode(given_expr)), " must be ", $(QuoteNode(op)), " ", $(QuoteNode(expected_expr)),
                   ". Expected: ", expected_val,
                   ". Given: ", val, ".")
        )
    end
end

macro assert_in(sym_expr, tuple_expr)
    quote
        sym_val = $(esc(sym_expr))
        tuple_val = $(esc(tuple_expr))
        @assert sym_val in tuple_val (
            string($(QuoteNode(sym_expr)), " must be one of ", $(QuoteNode(tuple_expr)),
                   ". Given: ", sym_val, ".")
        )
    end
end