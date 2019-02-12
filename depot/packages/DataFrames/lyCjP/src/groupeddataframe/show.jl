function Base.show(io::IO, gd::GroupedDataFrame;
                   summary::Bool = true)
    if allgroups
        for i = 1:N
        end
        if N > 0
            identified_groups = [string(parent_names[col], " = ", repr(first(gd[1][col])))
                                 for col in gd.cols]
            show(io, gd[1], summary=false,
                 allrows=allrows, allcols=allcols, rowlabel=rowlabel)
        end
        if N > 1
            identified_groups = [string(parent_names[col], " = ", repr(first(gd[N][col])))
                                 for col in gd.cols]
        end
    end
end
function Base.show(df::GroupedDataFrame;
                   summary::Bool = true) # -> Nothing
    return show(stdout, df,
                rowlabel=rowlabel, summary=summary)
end
